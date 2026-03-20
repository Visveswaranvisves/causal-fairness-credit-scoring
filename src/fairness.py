import pandas as pd
import matplotlib.pyplot as plt

def check_gender_bias(df):
    male = df[df['gender'] == 1]['target'].mean()
    female = df[df['gender'] == 0]['target'].mean()

    print("Male approval rate:", male)
    print("Female approval rate:", female)

    return male, female


def plot_bias(male, female):
    plt.bar(['Male', 'Female'], [male, female])
    plt.title("Gender Bias in Loan Approval")
    plt.show()