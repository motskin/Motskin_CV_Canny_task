import os

def is_image_file(filename:str):
    return str.lower(os.path.splitext(filename)[1]) in {'.tiff', '.tif', '.jpeg', '.jpg', '.jpe', '.bmp', '.png'}

def get_imagefiles_and_folders(folder: str):
    items = [os.path.join(folder, item) for item in os.listdir(folder)]     # получаем содержимое папки
    image_filenames = [file_name for file_name in items if os.path.isfile(file_name) if is_image_file(file_name)]  # файлы изображений внутри папки folder
    subfolders = [file_name for file_name in items if not os.path.isfile(file_name)]  # вложенные папки внутри папки folder
    return image_filenames, subfolders

def get_all_image_file_names_in_folder(folder:str):
    image_filenames, subfolders = get_imagefiles_and_folders(folder)   # получаем имена файлов изображений и папок, внутри папки folder
    result_list = list()        # основной список с именами файлов
    result_list.extend(image_filenames) # добавляем в основной список имена файлов, внутри папки folder

    for subfolder in subfolders:        # проходим по всем вложенным папкам
        result_list.extend(get_all_image_file_names_in_folder(subfolder))           # то размещаем имена файлов в основной папке
    
    return result_list      # возвращаем список имён всех файлов.
    