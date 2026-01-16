import json, time, math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import heapq
from tot_harness.voting import aggregate, extract_answer

@dataclass
class Node:
    node_id: str
    parent_id: Optional[str]
    depth: int
    prompt: str
    completion: str
    prm_score: float
    step_scores: List[float]
    generated_tokens: int

def _path_ids(nodes: Dict[str, Node], leaf_id: str) -> List[str]:
    out = []
    cur = leaf_id
    while cur is not None and cur in nodes:
        out.append(cur)
        cur = nodes[cur].parent_id
    return list(reversed(out))

def run_tot_one_problem(
    backend,
    token_counter,
    scorer,                    
    question_id: str,
    question: str,
    gold_answer: str,          
    cfg: Dict[str, Any],
    log_f
) -> Dict[str, Any]:
    dcfg = cfg["dpts_config"]
    lcfg = cfg["llm_config"]
    pcfg = cfg["prm"]

    tree_width = int(dcfg["tree_width"])
    tree_depth = int(dcfg["tree_depth"])
    max_rollout_iters = int(dcfg["max_rollout"])
    max_step_time_s = int(dcfg["max_step_time"])
    max_tokens_global = int(dcfg["max_new_tokens"])

    voting_method = dcfg.get("voting_method", "all")    # "all" -> majority vote (with fallback)

    sampling = dict(
        temperature=float(lcfg["temperature"]),
        top_p=float(lcfg["top_p"]),
        max_new_tokens=int(lcfg["max_new_tokens"]),
        do_sample=bool(lcfg.get("do_sample", True)),
        num_beams=int(lcfg.get("num_beams", 1)),
    )
    base_seed = int(cfg.get("seed", 12345))

    root_prompt = (
        "Please reason step by step. Put your final answer as '#### <number>'.\n\n"
        f"{question}\n\nSolution:\n"
    )

    t_start = time.time()

    nodes: Dict[str, Node] = {}
    root = Node("root", None, 0, root_prompt, "", float("-inf"), [], 0)
    nodes[root.node_id] = root
    
    # Best-first frontier: max PRM first (heapq is min-heap so we store -score)
    # tuple = (neg_priority, tie_breaker, node_id)
    frontier = []
    push_seq = 0

    # root should always expand first
    heapq.heappush(frontier, (float("-inf"), push_seq, "root")) 
    push_seq += 1

    

    # candidates collected for voting
    cand_ids: List[str] = []

    total_generated_tokens = 0
    total_llm_time = 0.0
    total_prm_time = 0.0

    next_id = 0
    expanded_children = 0

    stop_reason = "budget_rollout"
    for it in range(max_rollout_iters):
        if (time.time() - t_start) > max_step_time_s:
            stop_reason = "budget_time"
            break
        if total_generated_tokens >= max_tokens_global:
            stop_reason = "budget_tokens"
            break
        if not frontier:
            stop_reason = "frontier_empty"
            break

        # Expand up to tree_width parents this iteration 
        parents = []
        for _ in range(tree_width):
            if frontier:
                _, _, nid = heapq.heappop(frontier)
                parents.append(nid)

        log_f.write(json.dumps({
            "problem_id": question_id,
            "event": "rollout_iter",
            "iter": it,
            "parents": parents,
            "frontier_len_before": len(frontier) + len(parents),
            "total_generated_tokens_so_far": total_generated_tokens,
        }) + "\n")

        branch_factor = int(dcfg.get("num_branch", 1)) 

        for p_id in parents:
            if total_generated_tokens >= max_tokens_global: stop_reason = "budget_tokens"; break
            if (time.time() - t_start) > max_step_time_s: stop_reason = "budget_time"; break

            parent = nodes[p_id]
            if parent.depth >= tree_depth:
                
                continue

            parent_text = parent.prompt + parent.completion
            seed0 = base_seed + (hash(question_id) % 10_000_000) + it * 10000 + (hash(p_id) % 1000) * 10
                # LLM generate
            t0 = time.time()
            if hasattr(backend, "generate_n"):
                outs = backend.generate_n(parent_text, sampling, seed0, n=branch_factor)
            else:
                outs = []
                for j in range(branch_factor):
                    outs.append(backend.generate(parent_text, sampling, seed0 + j))
            t1 = time.time()
            if outs and all(getattr(o, "time_llm_forward_s", None) is not None for o in outs):
                total_llm_time += sum(o.time_llm_forward_s for o in outs)  # best-effort
            else:
                total_llm_time += (t1 - t0)


            # Generate *branch_factor* children from this same parent
            for j, res in enumerate(outs):
                if total_generated_tokens >= max_tokens_global:
                    stop_reason = "budget_tokens"
                    break

            

                completion = res.text
                gen_toks = token_counter.count_generated_tokens_delta(parent_text, completion)

                # dead nodes
                if gen_toks == 0 or (completion.strip() == ""):
                    log_f.write(json.dumps({
                        "problem_id": question_id,
                        "event": "dead_child",
                        "iter": it,
                        "parent_id": p_id,
                        "why": "empty_completion",
                    }) + "\n")
                    continue

                total_generated_tokens += gen_toks
                expanded_children += 1

                # PRM score
                s0 = time.time()
                prm_score = float(scorer.score(question, completion))
                s1 = time.time()
                total_prm_time += (s1 - s0)

                # sanitize NaN
                if not math.isfinite(prm_score):
                    prm_score = -1.0

                nid = f"n{next_id}"
                next_id += 1
                node = Node(
                    node_id=nid,
                    parent_id=p_id,
                    depth=parent.depth + 1,
                    prompt=parent_text,
                    completion=completion,
                    prm_score=prm_score,
                    step_scores=[],
                    generated_tokens=gen_toks,
                )
                nodes[nid] = node
                heapq.heappush(frontier, (-prm_score, push_seq, nid))
                push_seq += 1

                # candidate policy
                if extract_answer(completion):
                    cand_ids.append(nid)
                meta = getattr(res, "meta", None) or {}
                log_f.write(json.dumps({
                    "problem_id": question_id,
                    "event": "expand",
                    "iter": it,
                    "parent_id": p_id,
                    "node_id": nid,
                    "depth": node.depth,
                    "prm_score": prm_score,
                    "generated_tokens": gen_toks,
                    "sgl_prompt_tokens": meta.get("prompt_tokens"),
                    "sgl_completion_tokens": meta.get("completion_tokens"),
                    "sgl_cached_tokens": meta.get("cached_tokens"),
                }) + "\n")


    # If no candidates, fall back to best PRM among all non-root nodes, might change later . 
    if not cand_ids:
        all_ids = [k for k in nodes.keys() if k != "root"]
        if all_ids:
            best = max(all_ids, key=lambda i: nodes[i].prm_score)
            cand_ids = [best]

    # Build candidate texts and PRM score lists for voting
    cand_texts = [nodes[c].completion for c in cand_ids]
    cand_vlists = [nodes[c].step_scores for c in cand_ids]  

    final_text = aggregate(voting_method, cand_texts, cand_vlists)
    final_answer = extract_answer(final_text)
    gold = extract_answer(gold_answer)

    is_correct = (final_answer != "" and gold != "" and final_answer == gold)

    # Tokens kept: pick the candidate whose completion equals final_text (first match)
    chosen_id = None
    for cid in cand_ids:
        if nodes[cid].completion == final_text:
            chosen_id = cid
            break
    if chosen_id is None:
        chosen_id = cand_ids[0]

    kept_ids = _path_ids(nodes, chosen_id)
    tokens_kept = sum(nodes[i].generated_tokens for i in kept_ids if i != "root")
    tokens_pruned = max(0, total_generated_tokens - tokens_kept)


    summary = {
        "problem_id": question_id,
        "stop_reason": stop_reason,
        "rollout_iters_target": max_rollout_iters,
        "nodes_generated": expanded_children,
        "num_candidates": len(cand_ids),
        "final_method": voting_method,
        "final_answer": final_answer,
        "gold_answer": gold,
        "is_correct": bool(is_correct),

        "total_generated_tokens": int(total_generated_tokens),
        "tokens_kept_in_final_path": int(tokens_kept),
        "tokens_pruned": int(tokens_pruned),
    

        "time_total_s": float(time.time() - t_start),
        "time_llm_forward_s": float(total_llm_time),
        "time_prm_s": float(total_prm_time),
    }
    log_f.write(json.dumps({"problem_id": question_id, "event": "summary", **summary}) + "\n")
    return summary
