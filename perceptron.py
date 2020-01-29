epoc = 0

weights = np.array([0.0,0.0])
beta = np.array([0.0])
y = np.array([1,1,-1],)
x = np.array([[3.0,3.0],
    [4.0,3.0],
    [1.0,1.0]])

step_size = 1

# Make a prediction based on current weights
def predict(x_, y_, beta, weights):
    
    return np.multiply(np.array(list(map(lambda x: (x @ weights) + beta, x_))).flatten(), y_)

flag = len(x)
while flag > 0 and epoc < 100:
    epoc += 1
    non_equal_prediction = np.where(predict(x, y, beta, weights) <= y)[0]
    if len(non_equal_prediction) > 0:
        loc_i = int(np.random.choice(non_equal_prediction, 1))
        weights += x[loc_i] * y[loc_i] * step_size
        beta += y[loc_i] * step_size
    flag = len(np.where(predict(x, y, beta, weights) <= y)[0])
print(weights)
print(beta)
