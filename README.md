# MIXUP-BASED DEEP METRIC LEARNING APPROACHES FOR INCOMPLETE SUPERVISION

By [Luiz H. Buris](http://), [Fabio A. Faria](https://).

UNIFESP SJC -  Instituto de Ciência e Tecnologia

## Introdução
Propomos três novas abordagens no contexto da DML. Estamos particularmente interessados no NNGK devido à sua robustez e simplicidade. Como tal, apresentamos variantes que aproveitam o Mixup para lidar com o aprendizado de métrica em cenários de supervisão incompletos.


- https://github.com/LukeDitria/OpenGAN
- https://github.com/facebookresearch/mixup-cifar10
- https://kevinmusgrave.github.io/pytorch-metric-learning/

## Citation

If you use this method or this code in your paper, then please cite it:

```
@article{buris2022mixup,
  title={Mixup-based Deep Metric Learning Approaches for Incomplete Supervision},
  author={Buris, Luiz H and Pedronette, Daniel CG and Papa, Joao P and Almeida, Jurandy and Carneiro, Gustavo and Faria, Fabio A},
  journal={arXiv preprint arXiv:2204.13572},
  year={2022},
  url={https:https://arxiv.org/pdf/2204.13572.pdf},
}
```

## Organização do code

- `train_MbDML1-NNGK_Mixup.py`: Esta abordagem chamada $MbDML-1$ é uma simples combinação entre as funções de perdas das abordagens originais (NNGK e Mixup para compor a função de perda final desta abordagem. 
- `train_MbDML2_MixupNNGK.py` :  Esta abordagem chamada $MbDML-2$, as imagens $x_i$ e $x_j$ do conjunto de dados de treinamento e seus respectivos rótulos $y_i$ e $y_j$ dentro do lote são interpoladas pelo método Mixup, são passadas pela rede CNN pré-treinada e classificadas pelo classificador $NNGK$.
- `train_MbDML3_MixupNNGK_NNGK.py` :  Esta abordagem chamada $MbDML-3$, é a simples combinação linear entre a função de perda $MbDML-2$ e a função de perda original $NNGK$.
- `train_MbDML4_MbDML3+NST.py` : Esta abordagem chamada \textit{MbDML} 4 é a combinação da estratégia supervisionada proposta neste trabalho \textit{MbDML} 3 com adição de uma técnica semi-supervisionada da literatura chamada \textit{Noisy Student Training} (NST). 
- `train_NNGK.py` : Esta abordagem $NNGK$ é a original do artigo que foi proposto a melhoria.


## Requisitos e instalação.
- Python version 3.6
- A [PyTorch installation](http://pytorch.org/)
- A [Pytorch-Metric-Learning installation](https://kevinmusgrave.github.io/pytorch-metric-learning/#installation)
- pip install -r requirements.txt


## MbDML
Comandos para executar os scrypt python de cada abordagenns

```sh
python3 train_MbDML1-NNGK_Mixup.py --max_epochs=200 --name "CIFAR10-MbDML1-NNGK_Mixup" --scale_mixup 2 --alpha 1 --beta 1 --data_dir datasets/CIFAR100K10/train --test datasets/CIFAR100K10/Test --save_dir results/neighbour=200 --num_classes 100 --tsne_graph False --im_ext png --gpu_id 0 --input_size 32

```

```sh
python3 train_MbDML2_MixupNNGK.py --max_epochs=200 --name "CIFAR10-MbDML2_MixupNNGK" --scale_mixup 2 --alpha 1 --alpha 0 --data_dir datasets/CIFAR100K10/train --test datasets/CIFAR100K10/Test --save_dir results/neighbour=200 --num_classes 100 --tsne_graph False --im_ext png --gpu_id 0 --input_size 32

```

```sh
python3 train_MbDML3_MixupNNGK_NNGK.py --max_epochs=200 --name "CIFAR10-MbDML3_MixupNNGK_NNGK" --scale_mixup 2 --alpha 1 --beta 1 --data_dir datasets/CIFAR100K10/train --test datasets/CIFAR100K10/Test --save_dir results/neighbour=200 --num_classes 100 --tsne_graph False --im_ext png --gpu_id 0 --input_size 32

```

## Curvas Accuracy.

As figuras mostram as curvas de precisão de cada uma das abordagens supervisionadas propostas $MbDML1$, $MbDML2$ e $MbDML3$ em comparação com as abordagens originais NNGK e Mixup durante cada época do processo de treinamento e a cada $5 épocas no conjunto de teste, nas quatro bases de imagens adotadas neste experimento (CIfar10, CIfar100, MNIST e Flowers17).

CIFAR10 - Train    |  CIFAR10 - Test
:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/Cifar10-XL10_ACC_Train.png) |  ![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/Cifar10-XL10_ACC_Test.png) 

CIFAR100 - Train    |  CIFAR100 - Test
:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/Cifar100-XL10_ACC_Train.png) |  ![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/Cifar100-XL10_ACC_Test.png) 

MNIST - Train    |  MNIST - Test
:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/Mnist-XL10_ACC_Train.png) |  ![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/Mnist-XL10_ACC_Test.png) 

FLOWER17 - Train    |  FLOWER17 - Test
:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/Flower17-XL10_ACC_Train.png) |  ![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/Flower17-XL10_ACC_Test.png) 

## Diferentes espaço de caracteristicas.
Diferentes embeddings definidos por cada abordagem: (a) amostras no espaço de características definido por uma CNN pré-treinada, (b)
mesmas amostras projetadas em um kernel gaussiano, (c) amostras no espaço de características da CNN pré-treinada junto com o novo
amostras criadas pelo Mixup e (d) amostras no espaço de recursos pela combinação de NNGK e Mixup. Observe que, neste
No artigo, existem esses quatro tipos possíveis de espaços de características, portanto, as três abordagens propostas baseadas em Mixup (MbDML)
são algumas combinações dos espaços de recursos existentes.

![N|Solid](https://github.com/henriqueburis/ICIP2022/blob/main/fig/spaces_b.png?raw=true )

Como pode ser visto na figura, as classes são consistentemente muito melhor separadas pelo Mixup(NNGK)

CIFAR100   |   CIFAR100
:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/cifar10_tsne.gif) |  ![](https://github.com/henriqueburis/ICIP2022/blob/main/fig/cifar100_tsne.gif) 

## Comparison
Mean accuracies (%) and standard deviation (±) over ten runs using 10% of the training set. Similar and the most accurate results are highlighted.
![N|Solid](https://github.com/henriqueburis/ICIP2022/blob/main/fig/Mean%20accuracies.PNG?raw=true)
