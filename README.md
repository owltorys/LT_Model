訓練紀錄：https://docs.google.com/spreadsheets/d/1-6HBljTxE02mqzWjtj0xdm0ToujPTkN0vovTRDokUek/edit?usp=sharing

驗證集的數據是挑F1 score最高的那個epoch紀錄

微調程式碼(train.py)指令：python train.py --width-mult 模型規格 --epochs 總epoch數 --batch-size batch_size --lr 學習率 --save-dir 儲存資料夾 --num-workers 線程數 --data-root 資料集位置 --full-csv 整個資料集的csv檔 --k-folds 要切成幾份 --auto-pipeline --seed 種子碼(通常用預設就行了) --ema-decay ema --label-smoothing label smoothing

例：python train.py --width-mult 2.0 --epochs 60 --batch-size 128 --lr 0.001 --save-dir model/checkpoint/teacher_model --num-workers 4 --data-root Dowdwen_set_resized --full-csv Dowdwen_set_resized.csv --k-folds 5 --auto-pipeline --seed 24 --ema-decay 0.999

知識蒸餾程式碼(DK.py)指令：python DK.py --learning-rate 學習率 --epochs 總epoch數 --batch-size batch_size --alpha α --temperature T --teacher-width-mult teacher model的架構大小 --teacher-model-path 教師模型檔案位置 --save-dir 儲存資料夾 --data-root 資料集位置 --full-csv 整個資料集的csv檔 --k-folds 要切成幾份 --auto-pipeline --qat-start-epoch 第幾個epoch開始QAT 

例：python DK.py --learning-rate 0.001 --epochs 60 --batch-size 128 --alpha 2 --temperature 2 --teacher-width-mult 2 --teacher-model-path model/ntd/model_best_ntd1_1.pth --save-dir model/checkpoint/student_model --data-root DowDwen_set_resized --full-csv Dowdwen_set_resized.csv --k-folds 5 --auto-pipeline --qat-start-epoch 40

教師模型測試程式碼(testt.py)指令：python testt.py --model-path 教師模型位置 --data-root 資料集位置 --batch-size batch_size --num-workers 線程數

例：python testt.py --model-path model/ntd/model_best_ntd1_1.pth --data-root DowDwen_set_resized --batch-size 128 --num-workers 0

學生模型測試程式碼(tests.py)指令：python tests.py --model-path 學生模型位置 --data-root 資料集位置 --num-workers 線程數

例：python tests.py --model-path model/ntd/model_best_ntd1_1_1.pth --data-root DowDwen_set_resized --num-workers 0
