import matplotlib.pyplot as plt

#plot between two columns of a dataframe 
def plot_data(df, x_variable, y_variable, title):
    plt.plot(df[x_variable], df[y_variable])
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    plt.title(title)
    plt.show()

#plot a series
def plot_series(series):
    plt.figure(figsize=(12, 6))
    plt.plot(series, color='blue')
    plt.ylabel(f'{series.name}', fontsize=16)
    plt.show()


#plot of initial series
def init_plot(series):
    fig, ax = plt.subplots()

    ax.plot(series)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{series.name}')
    plt.xticks(
    [0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409, 439, 468, 498], 
    ['Jan 2019', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan 2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
    fig.autofmt_xdate()
    plt.tight_layout()


def init_plot2(series):
    fig, ax = plt.subplots()
    ax.plot(series)
    ax.set_xlabel('Time')
    ax.set_ylabel(series.name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()