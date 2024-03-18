# Understanding_prompts_via_corruption

# Requirements
```
pip install -r requiremensts.txt
```

# A. Create Prompts

We take original task files from Super-NaturalInstructions and create the baseline files for each tasks. We also add in-line instructions which are not a part of SupernaturalInstructions. We used PromptSource to do changes to the instructions and labelspace to make the prompt easy and understandable.

- **Get original task files** -  We copied the task files manually from Super-NaturalInstructions dataset files to `src/data_generator/original_tasks`.
  
- **Reorder samples** - We need to reorder the samples in the original tasks files such that the top 100 samples are balanced and can be used to evaluation. We do that by using this command. 
        
        python src/data_generator/reorder_instances_for_testing.py

The reordered datafiles are saved to `src/data_generator/reordered_original_tasks`.
        
- **Create baseline files** - Baseline files are the files with baseline corruptions. The baseline files are created for each task and the files have suffix `_baseline` and can be found at `src/data_generator/processed-tasks`. 
  
  Note: `_baseline` files includes all components. Instruction, inline instruction in each demostrations and demostration inputs and labels.

  To create baseline files, run

        python src/data_generator/create_baseline_data.py

- **Add Semantic corruptions** - run the following command to add semantic corruptions to the baseline file. The corruption name is added as suffix to the task files. The corrupted files are stored in `src/data_generator/processed_tasks`.

        python src/data_generator/create_corrupted_data.py


# B. Datasets, models and metrics

We evaluate 10 models on 10 datasets. Out of 10 tasks, 8 are classification tasks and 2 are generation tasks.

- Classification Datasets are 
1) task1344_glue_entailment_classification
2) task843_financial_phrasebank_classification_baseline
3) task1645_medical_question_pair_dataset_text_classification
4) task116_com2sense_commonsense_reasoning
5) task512_twitter_emotion_classification
6) task379_agnews_topic_classification
7)  task828_copa_commonsense_cause_effect
8)  task1346_glue_cola_grammatical_correctness_classification

- generation datasets are
1) task1564_triviaqa_answer_generation 
2) task835_mathdataset_answer_generation 

- Models are GPT2-xl, GPT-J-6B, Pythia-12B, OPT-30B, OPT-30B-IML-MAX2, OPT-66B, Vicuna-33B, Llama-7B, Llama-2-70B and Llama-2-70B-
chat.

- Evaluation metrics - Exact match for classification and Rouge-L for generation tasks.


# C. Evaluation

- The datasets for evaluation are listed in `src/evaluation/test_tasks.txt`.
- To evaluate the models run
        
        python evaluation/evaluate_model.py /
        --modelname [MODELNAME] /
        --corruptions_name_list [LIST OF CORRUPTION NAMES] /
        --task_set_no [SET NUMBER] /
        --max_target_len [MAXIMUM TARGET LENGTH] /
        --batch_size [EVALUATION BATCH SIZE]

        e.g. python evaluation/evaluate_model.py /
        --modelname facebook/opt-125m /
        --corruptions_name_list [empty,only_instruction] /
        --task_set_no set1 /
        --max_target_len 10 /
        --batch_size 16