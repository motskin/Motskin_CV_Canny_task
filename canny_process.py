import os
import gc
import cv2
import shutil
import time
import numpy as np
from multiprocessing import Pool, cpu_count

from utils.core import get_all_image_file_names_in_folder
from utils.progress_bar import ProgressBar


def error_process(exception: BaseException):
    print(exception)
    raise exception

def process_image(img_path:str):
    img = cv2.imread(img_path)
    img = cv2.blur(img, (3,3))
    # edges = cv2.Canny(img, 150, 200)
    # edges = cv2.Canny(img, 70, 164)
    # edges = cv2.Canny(img, 43, 86)
    edges = cv2.Canny(img, 130, 75)

    return edges

def main():
    input_folder = "/home/motskin/test_tasks/Regula_Motskin_CV_Canny/data/temp"

    output_folder = "/home/motskin/test_tasks/Regula_Motskin_CV_Canny/data/canny"
    chunk_size = 1000

    if os.path.exists(output_folder):      # очищаем выходную папку, если она существует или создаём
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # получаем пути ко всем файлам, которые нужно обрабатывать
    all_file_names = get_all_image_file_names_in_folder(input_folder)
    print(f"Изображений на обработку: {len(all_file_names)}")

    num_workers = int(cpu_count() - 2)
    print(f"Программа будет выполняться в {num_workers} потоках")

    chunks = np.array_split(list(all_file_names), max(1, int(len(all_file_names) / chunk_size)))
    print(f"Количество групп данных: {len(chunks)}")

    progress = ProgressBar(len(all_file_names), "Обработки: ")
    idx = 0

    start_time = time.time()

    with Pool(processes = num_workers) as pool:
        for chunk in chunks:
            task_dict = dict()   # для управления асинхронными тасками
            for file_name in chunk:
                key = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_name))[0]}.png")
                task_dict[key] = pool.apply_async(process_image, (file_name,), error_callback=error_process)

            for key in task_dict.keys():
                result_img = task_dict[key].get(timeout=20)
                cv2.imwrite(key, result_img)

                del result_img        # указываем, что эта память уже больше не нужна
                
                progress.update(idx)
                idx += 1

            del task_dict

            gc.collect()

    end_time = time.time()

    print(f"Операция длилась (сек): {int(end_time - start_time)} ")

if __name__ == '__main__':
    main()

    print("Программа завершена")