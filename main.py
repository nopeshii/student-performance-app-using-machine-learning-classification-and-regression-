print("================================")
print("Student Performance Prediction")
print("================================")

print("1. Dataset Exploration")
print("2. Regression")
print("3. Classification")
print("4. Model Comparison")
print("5. Graphs")

choice = input("Enter choice: ")
if choice == "1":
    import explore_data

elif choice == "2":
    import regression

elif choice == "3":
    import classification

elif choice == "4":
    import model_comparison

elif choice == "5":
    import graphs

elif choice == "6":
    import predict_score

else:
    print("Invalid Choice")
    print("6. Predict Student Score")