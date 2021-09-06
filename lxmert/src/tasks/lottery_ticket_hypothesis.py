import torch
import random
import numpy as np


def find_total_trainable_weight(vqa, not_pruned_layers):
    i = -1
    weights = list(vqa.model.state_dict().values())
    totalParams = 0
    for v in weights:
        i += 1
        if i in not_pruned_layers or i == len(weights) - 1:
            continue
        else:
            totalParams += int(np.prod(v.shape))
    
    return totalParams


def weight_prune_layer(k_weights, k_sparsity, mask):
    """
    Takes in matrices of kernel and bias weights (for a dense
      layer) and returns the unit-pruned versions of each
    Args:
      k_weights: 2D matrix of the 
      b_weights: 1D matrix of the biases of a dense layer
      k_sparsity: percentage of weights to set to 0
    Returns:
      kernel_weights: sparse matrix with same shape as the original
        kernel weight matrix
      bias_weights: sparse array with same shape as the original
        bias array
    """
    try :
      sorted_weights = np.sort(np.abs(k_weights[mask == 0]))

      # Determine the cutoff for weights to be pruned.
      cutoff_index = np.round(k_sparsity * sorted_weights.size).astype(int)
      cutoff = sorted_weights[cutoff_index]
      
      updated_mask = np.where(np.abs(k_weights) <= cutoff, np.ones(mask.shape), mask)
      
      k_weights[updated_mask == 1] = 0
    
    except:
      print("error")
      return k_weights, mask
    
    return k_weights, updated_mask


def low_magnitude_pruning(vqa, sparcity, args, log_result, file_name):
    logs = ""
    weights = vqa.model.state_dict()
    weights_key = list(weights.keys())
    not_pruned_layers = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
    totalParams = find_total_trainable_weight(vqa, not_pruned_layers)
    print("totalParams : ", totalParams)
    mask = {}

    for i in range(0, len(weights_key)): 
        if i not in not_pruned_layers and i != len(weights_key) - 1:
            mask[i] = np.zeros_like(weights[weights_key[i]].cpu().numpy())
    
    iteration = 0
    pruned_subnetwork_len = 0
    while(True):
        newWeightDict = {}
        for i in range(0, len(weights_key)):
            if i in not_pruned_layers or i == len(weights_key) - 1:
                # newWeightList.append(weights[i])
                newWeightDict[weights_key[i]] = weights[weights_key[i]]
            else:
                pruned_subnetwork_len -= np.sum(mask[i])
                kernel_weights, mask[i] = weight_prune_layer(weights[weights_key[i]].cpu().numpy(), 0.1, mask[i])
                newWeightDict[weights_key[i]] = torch.from_numpy(kernel_weights)
                pruned_subnetwork_len += np.sum(mask[i])


        vqa.model.load_state_dict(newWeightDict)
        score = vqa.evaluate(vqa.valid_tuple)* 100.
        print(f'Score after iteration {iteration}: {score}')
        logs += f'Score after iteration {iteration}: {score}\n'
        log_result[f'Score after iteration {iteration}'] = score 
        weights = newWeightDict.copy()
        iteration += 1
        print(f"pruned len {pruned_subnetwork_len} from total {totalParams}")
        print((pruned_subnetwork_len/totalParams) * 100)
        if(pruned_subnetwork_len > sparcity * totalParams):
          break
        if((pruned_subnetwork_len/totalParams) * 100 >= sparcity):
          print(f"finish pruning {sparcity} percent")
          break

    return weights, mask, logs, log_result

def apply_mask(weights, mask):
  weights_key = list(weights.keys())
  for i in range(len(weights_key)):
    if(i in mask):
      weights[weights_key[i]][mask[i] == 1] = 0
  return weights

def high_magnitude_pruning(good_mask_dict):
  bad_subnet = {}
  for good_mask in good_mask_dict:
    mask = good_mask_dict[good_mask]
    total_bad = int(np.sum(mask))
    total_good = int((1-mask).sum())
    if total_good > total_bad:
      bad_indices = np.argwhere(mask == 1).tolist()
      remaining_indices = random.sample(np.argwhere(mask == 0).tolist(), total_good - total_bad)
      bad_indices.extend(remaining_indices) # Remaining heads sampled from "good" heads.
    else:
      bad_indices = random.sample(np.argwhere(mask == 1).tolist(), total_good)
    head_mask = np.zeros_like(mask)
    for idx in bad_indices:
      try:
        head_mask[idx[0], idx[1]] = 1
      except IndexError:
        head_mask[idx[0]] = 1
    assert int(head_mask.sum()) == total_good
    bad_subnet[good_mask] = head_mask
  return bad_subnet



###############################################################################
#Random_Subnetwork

def get_random_mask(mask):
  
  updated_mask = np.zeros_like(mask)
  uniform_random = np.random.rand(*mask.shape)
  
  thereshold = np.sum(mask)/np.size(mask)
  updated_mask[uniform_random < thereshold] = 1

  return updated_mask

def get_random_subnet(good_mask_dict):
  random_subnet = {}
  for good_mask in good_mask_dict:
    mask = good_mask_dict[good_mask]
    random_subnet[good_mask] = get_random_mask(mask)
  return random_subnet