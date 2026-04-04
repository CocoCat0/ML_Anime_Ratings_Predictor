#import Libraries: 
from sklearn.cluster import KMeans

#important local files libraries:

kmeans = KMeans(n_clusters = 3, random_state = 42)
labels = kmeans.fit_predict(Xs)

df_clean["cluster"] = labels

#main function

def main():
        #load and preprocess the kaggle data
        #Reviews
        #Animations

        #conduct and EDA

        #initialize the model and train the model

        #outputs the data into visualizations

        #print an end statemen
#end of main statement

if __name__ == "__main__":
    main()