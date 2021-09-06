declare -a arr=("Low_Magnitude" "Random" "High_Magnitude")

for seed_ in {0..2..1}
do
    #finetuned
    output_finetune="models/vqa_lxmert_finetuned_seed${seed_}"

     CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
     python3 src/tasks/vqa.py \
     --train train,nominival --valid minival  \
     --llayers 9 --xlayers 5 --rlayers 5 \
     --loadLXMERTQA snap/pretrained/model \
     --batchSize 32 --optim bert --lr 5e-5 --epochs 4  \
     --tqdm --output ${output_finetune} 
    
    # #test-standard
     output_test_result="models/vqa_lxmert_finetuned_seed${seed_}"
    
     CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
     python3 src/tasks/vqa.py \
     --tiny --train train --valid ""  \
     --llayers 9 --xlayers 5 --rlayers 5 \
     --batchSize 32 --optim bert --lr 5e-5 --epochs 4 --sparsity -1\
     --tqdm --output ${output_test_result}  --test test --load models/vqa_lxmert_finetuned_seed${seed_}/BEST
    
    for pruning_mode in "${arr[@]}"
    do
        for number in {10..90..10}

        do
            # The name of this experiment.
            name=$2

            # Save logs and models under model; make backup.
            output="models/vqa_lxmert_pruning_seed${seed_}"

            # See Readme.md for option details.
            #pruning
            CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
                python3 src/tasks/vqa.py \
                --train train,nominival --valid minival  \
                --llayers 9 --xlayers 5 --rlayers 5 \
                --loadLXMERTQA snap/pretrained/model \
                --batchSize 32 --optim bert --lr 5e-5 --epochs 4  \
                --tqdm --output ${output} --seed ${seed_}  --sparsity ${number} --pruning --pruningmode ${pruning_mode} 

             # #test-standard
            output_test_result="models/vqa_lxmert_pruning_seed${seed_}"

            CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
            python3 src/tasks/vqa.py \
            --tiny --train train --valid ""  \
            --llayers 9 --xlayers 5 --rlayers 5 \
            --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
            --tqdm --output ${output_test_result}  --test test \
            --sparsity ${number}  --pruningmode ${pruning_mode} \
            --load models/vqa_lxmert_pruning_seed${seed_}/${pruning_mode}-${number}_percentage_retrain_BEST
        done

        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
            python3 src/experiment_result.py 
    done
done

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python3 src/experiment_result.py 

exit 0

