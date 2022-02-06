Regularized dropout for text classification

modified from https://github.com/monologg/JointBERT


bert-base-chinese from https://huggingface.co/bert-base-chinese/tree/main
```
./bert-base-chinese
./bert-base-chinese/config.json
./bert-base-chinese/pytorch_model.bin
./bert-base-chinese/tokenizer.json
./bert-base-chinese/tokenizer_config.json
./bert-base-chinese/vocab.txt
```

Training & Evaluation & Prediction

```bash
$ !python main.py --task iflytek --model_type bert --model_dir iflytek_model --do_train --do_eval --do_rdrop

$ python predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir iflytek_model
```


## Dependencies

- python>=3.6
- torch==1.6.0
- transformers==3.0.2
