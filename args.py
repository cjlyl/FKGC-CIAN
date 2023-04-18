import argparse
import os

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", default="data/NELL/", type=str)
	parser.add_argument("--device", default='cuda:0', type=str)
	parser.add_argument("--random_embed", default=False, type=str,
						help='using pre-trained embeddings or initiaizing embeddings')
	parser.add_argument("--few", default=5, type=int)
	parser.add_argument("--test", action='store_true')
	parser.add_argument("--embed_model", default='TransE', type=str)
	parser.add_argument("--max_neighbor", default=100, type=int)
	parser.add_argument("--hidden_size", default=100, type=int)

	parser.add_argument("-epo", "--epoch", default=100000, type=int)

	parser.add_argument("--n_layers_tf", default=4, type=int)
	parser.add_argument("--num_head", default=4, type=int)
	parser.add_argument("--dropout_input", default=0.2, type=float)
	parser.add_argument("--dropout_TF", default=0.3, type=float)

	parser.add_argument("--num_query", default=128, type=int)

	parser.add_argument("--batch_size", default=512, type=int)
	parser.add_argument("--use_pretrain", default=True, type=str)

	parser.add_argument("--fine_tune", action='store_true')

	parser.add_argument('--load_model', default=False, type=str, help='Load existing model?')
	parser.add_argument("--eval_every", default=10000, type=int)

	parser.add_argument("--train_few", default=1, type=int)
	parser.add_argument("--prefix", default='intial', type=str)
	parser.add_argument("--random_seed", default=1, type=int)

	parser.add_argument("--seed", default='19950902', type=int)
	parser.add_argument("--max_batches", default=300000, type=int)
	parser.add_argument("--weight_decay", default=0, type=float)

	parser.add_argument("--lr", default=5e-5, type=float)

	parser.add_argument("--symbol_embed_lr", default=5e-5, type=float) #fix
	parser.add_argument("--base_lr", default=5e-4, type=float)  #fix

	parser.add_argument("--warm_up_step", default=10000, type=int)
	parser.add_argument("--optimizer", type=str, default="Adam",
						help="Which optimizer to use?")
	parser.add_argument("--momentum", type=float, default=0.9)
	parser.add_argument("--margin", type=float, default=5.0,
						help="The margin between positive and negative samples in the max-margin loss") #original 5.0
	parser.add_argument("--grad_clip", default=5.0, type=float)
	parser.add_argument("--log_every", default=50, type=int)

	args = parser.parse_args()

	if not os.path.exists('models'):
		os.mkdir('models')
	args.save_path = 'models/' + args.prefix


	return args

