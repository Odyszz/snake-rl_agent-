# ОКРУЖЕНИЕ
### Награды и штрафы:
- смерть = удар в стену | удар в себя | Некорректное действие | превышение максимального количества шагов
- Смерть (удар в стену или удар в себя)________ -3.0
- Тайм-аут (превышено 200 шагов)_______________ -3.0
- Некорректное действие (разворот назад)_______ -2.0
- пропуск яблока которое можно было съесть_____ -1.5
- Съеденное яблоко ____________________________ 35 + длина змеи
- Победа (заполнение 25 клеток)________________ 2000
### Размер поля
- 5\5
### Действия:
1. up
2. down
3. left
4. right
### Старт:
- максимальное количество шагов 200
- центральные клетки `(3, 3), (3, 2), (3, 1)` 
- начальная длинна 3
- движение вверх
# ТЕХНИЧЕСКАЯ ЧАСТЬ
### Структура проекта
```
├── main.py              # основной файл
├── snake_env.py         # файл окружения
├── testing.py           # функции тестирования готовой модели
├── gif/                 # папка для сохранения побед в gif формате
├── src/        
│   ├── model.py         # модель
│   └── train.py         # функции памяти, определения действия и обучения и сохранения
├── test_snake.log       # статистика тестирования модель (появится после завершения обучения)  
├── train_snake.log       # статистика обученния модель (появится после запуска)  
├── snake_model.pth      # обученная модель (появится после завершения обучения)  
├── snake_model_best.pth # обученная модель (появится после запуска)  
└── README.md
```
### цель
Обучить агента (змею) выживать и набирать максимальное количество очков в игре Snake, используя Deep Q-Network (DQN).

### Данные фильтрация и статистика
- Данные собираются по мере обучения
- Фильтрация реализована косвенно за счёт deque(maxlen=100_000) неактуальные данные вытесняются
- #### Статистика ведётся в train_snake.log и test_snake.log а именно логируются: 
    - Номер эпизода
    - Длинна змеи
    - Общий счет
    - Причина окончания игры
    - Количество шагов

### Модель:
- DQN с несколькими слоями (SnakeDQN)

### Архитектура:
  ```Input(19) → Linear(196) → ReLU → Linear(324) → ReLU → Linear(256) → ReLU → Linear(4) ```
- Вход: 19 признаков (состояние игры)
- Выход: 4 Q-значения (Количество действий)

### Обучение
- 30 000 эпизодов
- Optimizer: AdamW
- weight_decay = 0.01
- LR = 0.0005
- BATCH_SIZE = 256
- Loss: MSE


### Результаты
- Средняя длина змеи: ~17.742
- Полное прохождение карты (25 клеток): ~1.6% (8 из 500)
- больше 17 очков в 60.8% игр
- больше 14 очков в 76.2% игр
- больше 10 очков в 85.2% игр


## Итог
Модель верно определяет положение на карте и успешно этим пользуется, набирая больше 13 очков в 82.2% игр, а так же проходя игру в 1.6% случаях
