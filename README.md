# ReStruct
Large Language Model-driven Meta-structure Discovery in Heterogeneous Information Network

## File description
`preprocess_recommendation.py`: Script that preprocesses data for recommendation.

`gen_neg_recommendation.py`: Script that preprocesses data for recommendation (generates negative samples).

`preprocess_node_classification.py`: Script that preprocesses data for node classification.

`data_recommendation/`: Preprocessed data for recommendation.

`data_node_classification/`: Preprocessed data for node classification.

`llm_component_no_prob.py`: LLM object.

`main_recommendation.py`: Script that searches for meta-structures for recommendation.

`main_node_classification.py`: Script that searches for meta-structures for node classification.

`model_recommendation.py`: Model for evaluation in recommendation.

`model_node_classification.py`: Model for evaluation in node classification.

`train_recommendation.py`: Script that evaluates discovered meta-structures for recommendation.

`train_node_classification.py`: Script that evaluates discovered meta-structures for node classification.

`llm_explanation.py`: Script that generates differential explanations for meta-structures.

`utils.py`: Script that contains useful functions.

## Preparing data
**We already provide the preprocessed data under `data_recommendation/` and `data_node_classification/`. **

## Search for meta-structures (example)
```shell
python main_recommendation.py --dataset Yelp --num_generations 30
python main_node_classification.py --dataset IMDB --num_generations 30
```

## Evaluation (example)
```shell
python train_recommendation.py --dataset Yelp --lr 0.01 --wd 0.002 --dropout 0.5  
python train_node_classification.py --dataset IMDB --lr 0.01 --wd 0.0002 --dropout 0.75 --no_norm 
```
