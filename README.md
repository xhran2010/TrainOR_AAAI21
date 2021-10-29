# TrainOR_AAAI21
This is the official implementation of our AAAI'21 paper:

>Haoran Xin, Xinjiang Lu, Tong Xu, Hao Liu, Jingjing Gu, Dejing Dou, Hui Xiong, **Out-of-Town Recommendation with Travel Intention Modeling**, In Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI’21), Online, 2021, 4529-4536.

both PaddlePaddle and Pytorch versions are provided.
> PaddlePaddle: https://www.paddlepaddle.org.cn \
Pytorch: https://pytorch.org


If you use our codes in your research, please cite:
```
@inproceedings{xin2021out,
  title={Out-of-Town Recommendation with Travel Intention Modeling},
  author={Xin, Haoran and Lu, Xinjiang and Xu, Tong and Liu, Hao and Gu, Jingjing and Dou, Dejing and Xiong, Hui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4529--4536},
  year={2021}
}
```

## Requirements
- Python 3.x
- Paddlepaddle 2.x / Pytorch >= 1.7

## Data Format
For check-in data, you need to format the hometown and out-of-town check-ins of users in two respective files following:
```
{user id}\t{timestamp}\t{poi id}\t{poi tag}
```
For POI distance data, please format as:
```
{poi id 1}\t{poi id 2}\t{distance}
```
Also, we provided a toy data generator to help you run the code. Run:
```
python generate_toy_data.py
```
to generate the toy data.

## Run Our Model
Simply run the following command to train:
```
cd ./PaddlePaddle
python run.py --ori_data {...} --dst_data {...} --dist_data {...} ---save_path {...} --mode train
```
Then, test the performance with a trained TrainOR model:
```
cd ./PaddlePaddle
python run.py --ori_data {...} --dst_data {...} --dist_data {...} --test_path {...} --mode test
```