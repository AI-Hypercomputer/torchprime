import torch


def convert_hf_state_dict_for_grouped_moe(hf_state_dict, config):
  """
  Converts a Hugging Face state_dict with per-expert weights in-place
  to use the grouped weight format.

  Args:
    hf_state_dict (dict): The state_dict from the Hugging Face model.
    config: The model configuration, used to get the number of experts.

  Returns:
    dict: The modified state_dict.
  """
  # Find all unique MoE layer prefixes (e.g., "model.layers.0.mlp.", "model.layers.1.mlp.", etc.)
  moe_prefixes = set()
  for key in hf_state_dict.keys():  # noqa: SIM118
    if "experts.0.gate_proj.weight" in key:
      # Assumes key format is like '...<prefix>.experts.0.gate_proj.weight'
      prefix = key.split("experts.0.gate_proj.weight")[0]
      moe_prefixes.add(prefix)

  if not moe_prefixes:
    print("No MoE layers with per-expert weights found to convert.")
    return hf_state_dict

  E = config.n_routed_experts

  print(f"Found and converting {len(moe_prefixes)} MoE layers with {E} experts each...")

  for prefix in moe_prefixes:
    # Pop all the old per-expert weights from the dictionary, transposing them
    w_g_list = [
      hf_state_dict.pop(f"{prefix}experts.{e}.gate_proj.weight").t() for e in range(E)
    ]
    w_u_list = [
      hf_state_dict.pop(f"{prefix}experts.{e}.up_proj.weight").t() for e in range(E)
    ]
    w_d_list = [
      hf_state_dict.pop(f"{prefix}experts.{e}.down_proj.weight").t() for e in range(E)
    ]

    # Stack them to create the new grouped tensors
    Wg = torch.stack(w_g_list, dim=0)
    Wu = torch.stack(w_u_list, dim=0)
    Wd = torch.stack(w_d_list, dim=0)

    # Add the new grouped weight keys to the dictionary
    hf_state_dict[f"{prefix}grouped.W_gate"] = Wg
    hf_state_dict[f"{prefix}grouped.W_up"] = Wu
    hf_state_dict[f"{prefix}grouped.W_down"] = Wd

    print(f"  - Converted weights for prefix: {prefix}")

  return hf_state_dict


def revert_grouped_moe_to_hf_state_dict(
  grouped_state_dict, n_routed_experts, keep_existing_grad: bool = True
):
  """
  Converts a state_dict with grouped MoE weights back into the standard HF format.

  The returned tensors are always detached from the computation graph.

  Args:
    grouped_state_dict (dict): The state_dict with keys like '...grouped.W_gate'.
    n_routed_experts: The number of experts used in the model.
    keep_existing_grad (bool, optional): If True, any existing .grad attribute
      on a tensor will be copied to the new, detached tensor. If False,
      the .grad attribute will be discarded. Defaults to True.

  Returns:
    dict: The modified state_dict with per-expert, detached keys.
  """
  moe_prefixes = set()
  for key in list(grouped_state_dict.keys()):
    if key.endswith("grouped.W_gate"):
      prefix = key.split("grouped.W_gate")[0]
      moe_prefixes.add(prefix)

  if not moe_prefixes:
    print("No grouped MoE layers found to revert.")
    return grouped_state_dict

  E = n_routed_experts
  print(f"Found and reverting {len(moe_prefixes)} MoE layers to {E} experts each...")

  for prefix in moe_prefixes:
    Wg = grouped_state_dict.pop(f"{prefix}grouped.W_gate")
    Wu = grouped_state_dict.pop(f"{prefix}grouped.W_up")
    Wd = grouped_state_dict.pop(f"{prefix}grouped.W_down")

    for e in range(E):
      # --- Process Gate Weight ---
      # 1. Slice and transpose the parameter tensor, then detach it.
      new_gate_tensor = Wg[e].t().detach()
      # 2. Check for grad on the PARENT tensor (Wg).
      if keep_existing_grad and Wg.grad is not None:
        # 3. Slice the PARENT's grad, transpose it, and assign it.
        new_gate_tensor.grad = Wg.grad[e].t().detach().clone()
      grouped_state_dict[f"{prefix}experts.{e}.gate_proj.weight"] = new_gate_tensor

      # --- Process Up Weight ---
      new_up_tensor = Wu[e].t().detach()
      if keep_existing_grad and Wu.grad is not None:
        new_up_tensor.grad = Wu.grad[e].t().detach().clone()
      grouped_state_dict[f"{prefix}experts.{e}.up_proj.weight"] = new_up_tensor

      # --- Process Down Weight ---
      new_down_tensor = Wd[e].t().detach()
      if keep_existing_grad and Wd.grad is not None:
        new_down_tensor.grad = Wd.grad[e].t().detach().clone()
      grouped_state_dict[f"{prefix}experts.{e}.down_proj.weight"] = new_down_tensor

    print(f"  - Reverted weights for prefix: {prefix}")

  return grouped_state_dict
