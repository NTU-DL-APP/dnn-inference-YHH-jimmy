import model_test

# 執行特定測試
model_test.test_inference()
model_test.test_softmax()
model_test.test_relu()

# 檢查準確度
acc = model_test.load_test_acc()
print(f"模型準確度: {acc}")
