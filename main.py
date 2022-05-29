import plot_map
from app import App


if __name__ == '__main__':
    model = App()
    df = model.process_all()
    reg, reg_cat = model.train(df)
    model.test(df, reg, reg_cat)
    plot_map.plot_map()
