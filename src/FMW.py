import numpy as np


def calculate_integral_image(mask):
    return np.cumsum(np.cumsum(mask, axis=0), axis=1)


def max_true_area(integral_image, width=600, height=420):
    rows, cols = integral_image.shape
    max_count = 0
    best_position = (0, 0)

    for i in range(rows - height + 1):
        for j in range(cols - width + 1):
            total_true = integral_image[i + height - 1, j + width - 1]
            if i > 0:
                total_true -= integral_image[i - 1, j + width - 1]
            if j > 0:
                total_true -= integral_image[i + height - 1, j - 1]
            if i > 0 and j > 0:
                total_true += integral_image[i - 1, j - 1]

            if total_true > max_count:
                max_count = total_true
                best_position = (i, j)

    return best_position, max_count


# 示例使用
if __name__ == '__main__':
    import dataset_util as dsutil
    dp_img = dsutil.read_depth_prior('/mnt/d/datasets/gated2depth/data/real/', '00020')
    # mask = np.random.randint(0, 2, (1080, 1920))  # 假设的掩码图
    mask = np.int16(dp_img[0])
    integral_image = calculate_integral_image(mask)
    position, count = max_true_area(integral_image)
    print("最佳位置:", position, "包含最多的true数量:", count)
