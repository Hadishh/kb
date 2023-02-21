output_path = "/home/hadisheikhi77/exp/output_wic_extended_v3.txt"
test_path = '/home/hadisheikhi77/exp/test/test'
model_archive = "/home/hadisheikhi77/exp/wic_extended_v3_full/model.tar.gz"

from bin.write_wic_for_codalab import write_for_official_eval
write_for_official_eval(model_archive, test_path, output_path)


f1 = open(output_path, 'r')
lines_pred = f1.readlines()
lines_pred = [i[0] for i in lines_pred]

f2  = open(test_path + ".gold.txt", 'r')
lines_true = f2.readlines()
lines_true = [i[0] for i in lines_true]
count = 0

for y_pred, y_true in zip(lines_pred, lines_true):
    if y_true == y_pred:
            count += 1
print(count)
print(f"accuracy: {count / len(lines_true)}")