import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

def draw_train_statistics(data: dict) -> plt.figure:
    """ Рисуем графики обучения на matplotlib.figure
        :param data: статистика обучения по эпохам. Представляет собой словарь где ключ - это номер эпохи,
                     а значение содержит другой словарь в котором находятся значения точности и потерь как для train так и для валидации
        :return: Возвращает plt.figure на которой изображены графики обучения
    """
    df = pd.DataFrame.from_dict(data, orient='index')    # переводим статистику в pandas.Dataframe (orient='index' позволяет правильно повернуть данные)

    # общее конфигурирование графиков
    fig, (ax_loss, ax_val_score) = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))        # создаём холст и 2 графика, в виде колонок 
    fig.suptitle('Training graphs', fontsize=20)                   # Подписываем всё изображение

    ax_loss.set_xlabel('Epoch')                                      # Подписываем оси x
    ax_val_score.set_xlabel('Epoch')                                 # на двух графиках
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))       # для двух графиков, указываем, что ось x 
    ax_val_score.xaxis.set_major_locator(MaxNLocator(integer=True))  # должна отображать целые числа

    # Прорисовка графики потерь
    min_train_loss = df["train_loss"].min()                                                           # Получаем лучшее значение валидационной точности
    min_train_epoch = df["train_loss"].idxmin()                                                      # Получаем номер эпохи, на которой была достигнута наивысшая точность
    ax_loss.title.set_text(f'Train loss: {min_train_loss:.5f}%, Epoch {min_train_epoch})')     # Подписываем график точности, дополнительно выводим значения
    ax_loss.set_ylabel('Loss')                                                               # Подписываем ось y как точность (сознательно на английском)
    df.plot(y = ['train_loss'], ax=ax_loss)                                            # отображаем 2 линии точности для Обучения и Валидации
    ax_loss.hlines(min_train_loss, 1, len(data.keys()), colors="r", linestyle='dashed')             # Рисуем горизонтальную линию по максимальной точности
    ax_loss.plot(min_train_epoch,min_train_loss,'ro')                                                 # Ставим на графике точку где была достигнута максимальная точность
    ax_loss.grid(color='g', linestyle='-', linewidth=1)                         # отображаем сетку

    # Прорисовка графика val score
    max_val_dice_coef = df["val_dice_coef"].max()                                                         # Получаем значение лучшей потери (минимальной) для валидации
    max_val_dice_epoch = df["val_dice_coef"].idxmax()                                                     # Получаем номер эпохи, на которой была достигнута минимальная ошибка
    ax_val_score.title.set_text(f'Dice coef: {max_val_dice_coef:.3f}, Epoch {max_val_dice_epoch})')     # Подписываем график потерь, дополнительно выводим значения
    ax_val_score.set_ylabel('Score')                                                                  # Подписываем ось Y как Потеря (Сознательно на английском)
    df.plot(y = ["val_dice_coef"], ax=ax_val_score)                                         # Рисуем 2 графика потерь для Обучения и Валидации
    ax_val_score.hlines(max_val_dice_coef, 1, len(data.keys()), colors="r", linestyle='dashed')           # Рисуем горизонтальную линию по минимальному значению ошибки
    ax_val_score.plot(max_val_dice_epoch,max_val_dice_coef,'ro')                                               # Ставим на графике точку, где была достигнута минимальная ошибка
    ax_val_score.grid(color='g', linestyle='-', linewidth=1)                        # отображаем сетку

    return fig              # возвращает холст, на котором выполняли рисования (Внимание! этот холст зарегистрировал в глобальном plt [смотри plt.close()])


def plot_img_and_mask(img, mask, canny_img):
    fig, ax = plt.subplots(1, 3 if canny_img is not None else 2, figsize=(18,6))
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Net Canny')
    ax[1].imshow(mask, cmap="gray")
    if canny_img is not None:
        ax[2].set_title('Real Canny')
        ax[2].imshow(canny_img, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.show()