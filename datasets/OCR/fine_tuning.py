import os

from vietocr.tool.config import Cfg

# dataset = 'ground_truth'
#
# txt_files = [os.path.join(dataset, file) for file in os.listdir(dataset) if file.endswith('.txt')]
# ground_truth = {}
# for file in sorted(txt_files):
#     with open(file, 'rb') as f:
#         s = f.read()
#         filename = os.path.splitext(os.path.splitext(os.path.split(file)[1])[0])[0]
#         ground_truth[filename] = s.decode('utf-8')
#
#
# gt_items = list(ground_truth.items())
# random.shuffle(gt_items)
#
# train_size = 0.8
#
# train_data = []
# test_data = []
# for i, (filename, annotation) in enumerate(gt_items):
#     if i < int(train_size * len(gt_items)):
#         train_data.append(''.join([os.path.join('processed', filename + '.jpg'), '\t', annotation, '\n']))
#         train_data = list(filter(('\n').__ne__, train_data))
#
#     else:
#         test_data.append(''.join([os.path.join('processed', filename + '.jpg'), '\t', annotation, '\n']))
#         test_data = list(filter(('\n').__ne__, test_data))
#
# with open('train_annotation.txt', 'wb') as f:
#     for row in train_data:
#         f.write(row.encode())
#     f.close()
#
# with open('test_annotation.txt', 'wb') as f:
#     for row in test_data:
#         f.write(row.encode())
#     f.close()
from config import PROJECT_ROOT

config = Cfg.load_config_from_name('vgg_transformer')

dataset_params = {
    'name': 'coop',
    # 'data_root': '../VietOCR/data_line',
    'data_root': '.',
    'train_annotation': 'train_annotation.txt',
    'valid_annotation': 'test_annotation.txt'
}

params = {
    'print_every': 200,  # hiển thị loss mỗi 200 iteration
    'valid_every': 10000,  # đánh giá độ chính xác mô hình mỗi 10000 iteraction
    'iters': 20000,
    'checkpoint': os.path.join(PROJECT_ROOT, 'weights/transformerocr_checkpoint.pth'),
    'export': os.path.join(PROJECT_ROOT, 'weights/transformerocr_finetuned.pth'),
    'metrics': 10000,
    'batch_size': 16
}
config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'

from vietocr.model.trainer import Trainer

trainer = Trainer(config, pretrained=True)

# sử dụng lệnh này để visualize tập train, bao gồm cả augmentation
trainer.visualize_dataset()

# # bắt đầu huấn luyện
trainer.train()
#
# # visualize kết quả dự đoán của mô hình
trainer.visualize_prediction()
#
# # huấn luyện xong thì nhớ lưu lại config để dùng cho Predictor
trainer.config.save('config_transformers.yml')
