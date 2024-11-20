#!/usr/bin/env python3
import csv
import random

import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class Perceptron:
    def __init__(
        self,
        learning_rate=0.01,
        activation_function=sigmoid,
        max_training_iterations=10**5,
    ):
        self._bias = None
        self._weights = None
        self._learning_rate = learning_rate
        self._max_iterations = max_training_iterations
        self._activation_function = activation_function

    def fit(
        self,
        predictors,
        expected_predictions,
    ):
        self._bias = random.uniform(-1, 1)
        self._weights = numpy.random.uniform(
            low=-1,
            high=1,
            # take number of weight must be equal to the number of predicting variables
            size=predictors.shape[1],
        )

        for _ in range(self._max_iterations):
            prediction = self.predict(predictors)
            errors = (expected_predictions - self.predict(predictors)).reshape(-1, 1)
            self._weights += (self._learning_rate * errors * predictors).sum(axis=0)
            self._bias += self._learning_rate * errors.sum()


    def predict(
        self,
        predictors,
    ):
        similarity = self._activation_function(
            predictors.dot(self._weights) + self._bias,
        )
        return numpy.round(similarity)


def main():
    with open('iris.csv') as csv_file:
        reader = csv.reader(
            csv_file,
            delimiter=',',
        )
        setosa_or_versicolor_rows = []
        for row in reader:
            if row[4] == 'setosa':
                setosa_or_versicolor_rows.append(
                    [
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        0,
                    ],
                )
            elif row[4] == 'versicolor':
                setosa_or_versicolor_rows.append(
                    [
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        1,
                    ],
                ) 
        random.shuffle(setosa_or_versicolor_rows)

        training_data_length = round(0.8 * len(setosa_or_versicolor_rows))
        training_data = setosa_or_versicolor_rows[:training_data_length]
        predictors = numpy.array([observation[:4] for observation in training_data])
        expected_predictions = numpy.array([observation[4] for observation in training_data])
        iris_perceptron = Perceptron()
        iris_perceptron.fit(
            predictors,
            expected_predictions,
        )

        validation_data = setosa_or_versicolor_rows[training_data_length:]
        for observation in validation_data:
            prediction = iris_perceptron.predict(
                numpy.array(observation[:4])
            )
            actual = observation[4]
            print(f'Perceptron predicted {prediction} while actual was {actual}')


if __name__ == '__main__':
    main()

