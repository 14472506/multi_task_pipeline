import json

def find_lowest_index_between_50_and_80(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    val_total = data.get('val_loss', [])
    if len(val_total) < 80:
        raise ValueError('The list "val_total" has less than 80 elements.')

    lowest_index = 50
    lowest_value = val_total[50]

    for index in range(51, 80):
        if val_total[index] < lowest_value:
            lowest_value = val_total[index]
            lowest_index = index

    lower_lowest_val = 2

    for index2 in range(0, 50):
        if val_total[index2] < lower_lowest_val:
            lower_lowest_val = val_total[index2]
            lowest_index = index

    return lower_lowest_val, lowest_value

if __name__ == '__main__':
    json_file = 'outputs/reduced_dataset_ssl/full_reduction_rotnet_pt_mask_rcnn_10/log.json'
    result, val = find_lowest_index_between_50_and_80(json_file)
    print(result, val)