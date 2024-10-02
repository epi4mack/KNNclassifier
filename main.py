import pandas as pd
from sklearn.preprocessing import LabelEncoder
from KNNclassifier import KNNclassifier

if __name__ == '__main__':
    data = pd.read_excel('Dataset.xlsx')

    data = data.drop(columns=[
        'Как часто вы берете инициативу в свои руки? / Баллы',
        'Как часто вы пропускаете завтраки? / Баллы',
        'Какая культура ближе / Баллы',
        'Выпиваете алкоголь / Баллы',
        'Любимое время года? / Баллы',
        'Что пьют родители / Баллы',
        'Какие напитки любите / Баллы',
        'Набрано баллов',
        'Всего баллов',
        'Результат теста'
        ])

    for column in list(data.columns):
        if column not in ('Возраст', 'Сколько спите ночью в среднем', 'Время подъема'):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])

    k = 1
    knn_classifier = KNNclassifier(k)
    knn_classifier.fit(data)

    size = len(data)
    correct = 0
    for i in range(size):
        test_point = list(data.loc[i])
        actual_preference = 'Чай' if test_point.pop(3) else 'Кофе'
        prediction = knn_classifier.predict(test_point)

        if prediction == actual_preference:
            correct += 1
            marker = '+'
        else:
            marker = '-'

        print(f'Для точки {i+1} предсказано предпочтение: {prediction} ({marker});')

    print(f'\nДанная модель имеет точность в {correct / size * 100:.2f}% при k={k}. Тестирование произведено на тренировочных данных.')
