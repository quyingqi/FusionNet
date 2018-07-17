import os, sys, math, random, logging, argparse, time, json
import utils

args = utils.add_argument()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'

import torch
#from logger import Logger
from evaluate import evalutate
from predict import predict_answer
from model import FusionNetReader


# set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.device >= 0:
    torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)

word_dict, pos_dict, ner_dict = torch.load(open(args.dict_file, 'rb'))

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, loc: storage)
    print('load model from %s' % args.resume_snapshot)
else:
    model = FusionNetReader(word_dict, vars(args), [pos_dict, ner_dict], [args.pos_vec_size, args.ner_vec_size])
    if args.word_vectors != 'random':
        model.embedding.load_pretrained_vectors(args.word_vectors, args.embedding_cache, binary=True, normalize=args.word_normalize)

# set model dir
model_folder, model_prefix = utils.get_folder_prefix(args, model)
log_file = os.path.join(model_folder, args.log_file)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

# setup tensorboard logger
#tb_log_dir = os.path.join(model_folder, 'tb_logs')
#tb_log = Logger(tb_log_dir)

if args.device >= 0:
    model.cuda(args.device)
if args.multi_gpu:
    model = torch.nn.DataParallel(model)

def get_data_dict(args, pt_file):
    data = torch.load(open(pt_file, 'rb'))
    data.set_batch_size(args.batch)
    data.set_device(args.device)
    return data

print('loading training data...')
#baidu_data = get_data_dict(args, args.baidu_data)
train_data = get_data_dict(args, args.train_data)
valid_data = get_data_dict(args, args.valid_data)

log.info('[Data loaded.]')

params = list()
for name, param in model.named_parameters():
    log.info('%s - %s' % (name, param.size()))
    params.append(param)

optimizer = getattr(torch.optim, args.optimizer)(params, lr=args.lr, weight_decay=args.regular_weight)

global_step = 0

def eval_epoch(_model, _data):
    _model.eval()
    answer_dict_old, acc_s, acc_e, acc = predict_answer(_model, _data)
    q_level_p_old, char_level_f_old = evalutate(answer_dict_old)
    return q_level_p_old, char_level_f_old, acc_s, acc_e, acc

def train_epoch(_model, _data):
    global global_step
    _model.train()
    loss_acc = 0
    num_batch = len(_data) / args.batch
    batch_index = 0
    forward_time = 0
    data_time = 0
    backward_time = 0
    back_time = time.time()
    for batch in _data.next_batch(ranking=False):
        batch_index += 1
        global_step += 1
        data_time = time.time() - back_time
        optimizer.zero_grad()

        start_time = time.time()
        loss = _model(batch)
        end_time = time.time()
        forward_time += end_time - start_time
        loss.backward()
        loss_acc += loss.data.item()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        optimizer.step()
        back_time = time.time()
        backward_time += back_time - end_time

        if batch_index % args.log_per_updates == 0:
#            print("iter: %d  %.2f  loss: %f" %(batch_index, batch_index/num_batch, loss.data[0]))
            log.info("iter: %d  %.2f  loss: %f" %(batch_index, batch_index/num_batch,
                                                  loss.data.item()))

#            tb_log.scalar_summary('train/loss', loss.data[0], global_step)

#    print(forward_time, data_time, backward_time)
    return (loss_acc / num_batch)


best_epoch = 0
log.info('[Training .]')
best_loss = 200.
best_cf = 0.
best_qp = 0.

for iter_i in range(args.epoch):
    log.warning('Epoch {}'.format(iter_i))
    # train
    start = time.time()
#    train_loss = train_epoch(model, baidu_data)
    train_loss = train_epoch(model, train_data)
    train_end = time.time()

    # eval dev
    q_p_old, c_f_old, acc_s, acc_e, acc = eval_epoch(model, valid_data)
    eval_end = time.time()

    train_time = train_end - start
    eval_time = eval_end - train_end

    iter_str = "Iter %s" % iter_i
    time_str = "%s | %s" % (int(train_time), int(eval_time))
    train_loss_str = "Loss: %.2f" % train_loss
    acc_result = "Acc: %.2f Acc_s: %.2f Acc_e: %.2f" %(acc, acc_s, acc_e)
    eval_result_old = "Query Pre: %.2f: Char F1: %.2f" % (q_p_old, c_f_old)
    log_str = ' | '.join([iter_str, time_str, train_loss_str, acc_result, eval_result_old])

    log.info(log_str)

#    tb_log.scalar_summary('dev/em', q_p_old, global_step)
#    tb_log.scalar_summary('dev/f1', c_f_old, global_step)
#    tb_log.scalar_summary('dev/acc', acc, global_step)

    # save
    if model_prefix is not None:
        if best_loss > train_loss:
            torch.save(model, model_prefix + '.best.loss.model')
            best_loss = train_loss
        if best_cf < c_f_old:
            torch.save(model, model_prefix + '.best.char.f1.model')
            best_cf = c_f_old
        if best_qp < q_p_old:
            torch.save(model, model_prefix + '.best.query.pre.model')
            best_qp = q_p_old

log.info("Best Train Loss: %s\n" % best_loss)
log.info("Best Char F1   : %s\n" % best_cf)
log.info("Best QUery Pre : %s\n" % best_qp)

