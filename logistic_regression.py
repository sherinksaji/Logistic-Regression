import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt

pathToTrain = "HW3_data/HW2_data/train_diabetes.csv"
df_train = pd.read_csv(
    pathToTrain,
    header=None,
    names=[
        "y",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x16",
        "x17",
        "x18",
        "x19",
        "x20",
    ],
)
df_train.reset_index(drop=True, inplace=True)  # Reset the index

print(df_train)


def linear_combi(theta, x_vector, theta_0):
    return np.dot(theta, x_vector) + theta_0


def log_likelihood_t(true_y, theta, x_vector, theta_0):
    s = true_y * linear_combi(theta, x_vector, theta_0)
    likelihood_t = math.exp(s) / (1 + math.exp(s))
    return math.log(likelihood_t)


def log_likelihood(df, theta, theta_0):
    x_columns = [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x16",
        "x17",
        "x18",
        "x19",
        "x20",
    ]

    df["log_likelihood"] = df.apply(
        lambda row: log_likelihood_t(
            row["y"], theta, np.array(row[x_columns]), theta_0
        ),
        axis=1,
    )
    return df["log_likelihood"].sum()


def error_t(true_y, theta, x_vector, theta_0):
    p_yi_given_xi = log_likelihood_t(true_y, theta, x_vector, theta_0)
    return math.log(p_yi_given_xi)


def delta_error_t_wrt_theta(true_y, theta, x_vector, theta_0):
    numerator = -true_y * x_vector
    denominator = 1 + math.exp(true_y * (np.dot(theta, x_vector) + theta_0))
    return numerator / denominator


def delta_error_t_wrt_theta_0(true_y, theta, x_vector, theta_0):
    numerator = -true_y
    denominator = 1 + math.exp(true_y * (np.dot(theta, x_vector) + theta_0))
    return numerator / denominator


def weight_update(true_y, theta, x_vector, theta_0, learning_rate):
    theta = theta - learning_rate * delta_error_t_wrt_theta(
        true_y, theta, x_vector, theta_0
    )
    theta_0 = theta_0 - delta_error_t_wrt_theta_0(true_y, theta, x_vector, theta_0)
    return theta, theta_0


def one_training_iteration(df, theta, theta_0, k):
    # print(df.shape[0])
    t = random.randint(0, df.shape[0] - 1)
    # print("t+1:",t+1)
    df_row_list = df.loc[t, :].values.flatten().tolist()
    y_t = df_row_list.pop(0)
    # print("y_t:",y_t)
    if len(df_row_list) > 20:
        df_row_list.pop(-1)
    x_t = np.array(df_row_list)
    # print("x_t:",x_t)
    theta, theta_0 = weight_update(y_t, theta, x_t, theta_0, k)
    # print("theta:",theta)
    # print("offset:",offset)
    return theta, theta_0


theta = np.zeros(20)
theta_0 = 0
number_of_iterations = 10000
k = 0.1
random.seed(1)
log_likelihood_ls = []
theta_store = []
plot_x_values = []
count = 0

while count < number_of_iterations:
    theta, theta_0 = one_training_iteration(df_train, theta, theta_0, k)
    count += 1
    if count % 100 == 0:
        theta_store.append(theta)

        log_likelihood_ls.append(log_likelihood(df_train, theta, theta_0))

        plot_x_values.append(count)

print(len(log_likelihood_ls))
print(theta)

# Plot y against x
plt.plot(plot_x_values, log_likelihood_ls, linestyle="-", color="b")

# Add labels and title
plt.xlabel("Number of iterations")
plt.ylabel("Log likelihood")
plt.title("Log likelihood vs. Number of iterations")

# Display the plot
plt.show()
