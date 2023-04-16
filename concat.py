import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():
    final_x = pd.DataFrame()
    final_y = pd.DataFrame()
    for i in range(1, 19):
        x = pd.read_csv("processing/features/" + str(i) + ".csv")
        y = pd.read_csv("processing/final_labels/" + str(i) + "yf.csv")

        final_x = pd.concat([final_x, x], axis=0)
        final_y = pd.concat([final_y, y], axis=0)

    scaler = MinMaxScaler().set_output(transform='pandas')
    final_x = scaler.fit_transform(final_x)

    final_x.to_csv("x.csv", index=False)
    final_y.to_csv("y.csv", index=False)
        
        
if __name__ == "__main__":
    main()
