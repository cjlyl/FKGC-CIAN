python main.py --fine_tune --lr 2e-4 --prefix nell_mainlr2e-4.5shot --device cuda:0
python main.py --fine_tune --lr 3e-4 --prefix nell_mainlr3e-4_1.5shot --device cuda:1



python main.py --fine_tune --few 5 --prefix nell_main_lr5e-4SElr5e-5.5shot
python main.py --fine_tune --few 3 --lr 4e-5 --prefix nell_main_lr5e-4SElr5e-5.3shot

python main.py --fine_tune --lr 2e-4 --prefix nell_wikilr2e-4SElr5e-5BaseLr5e-4.5shot --device cuda:1
python main.py --fine_tune --lr 4e-4 --prefix nell_wikilr4e-4SElr5e-5BaseLr5e-4.5shot --device cuda:0
