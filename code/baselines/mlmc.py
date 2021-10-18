from utils import get_stats


def main():
    depths = []
    mlmc = []
    # open csv
    with open('mlmc.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(',')
            depths.append(float(tokens[0]))
            mlmc.append(float(tokens[1]))

    correlation, rmse, p_val = get_stats(depths, mlmc)
    print(f"Pearson's Correlation coefficient: {correlation}, RMSE: {rmse}, p value: {p_val}")


if __name__ == "__main__":
    main()