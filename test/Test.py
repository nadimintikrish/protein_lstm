from utilities.DataPreProcessor import DataPreProcessor

test = DataPreProcessor()
test_X = [[[1, 2, 3], [2, 4, 5], [5, 6, 7]], [[1, 5, 6], [7, 8, 9], [0, 1, 2], [4, 6, 2]], [[1, 4, 1], [2, 4, 5]],
          [[0, 2, 5]], [[0, 5, 5], [1, 1, 1]]]
test_y = [[1, 0], [0, 1], [1, 0], [0, 1], [0, 1]]

for t1, t2 in test.batch_gen(test_X, test_y, 2):
    print(t1, t2)
