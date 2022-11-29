from abc import ABCMeta, abstractmethod
import numpy as np

eta = 500 #振動の時の終了回数終了回数(回数は適当)

class Clustering(metaclass=ABCMeta):
    k: int
    d: int
    centroids: np.ndarray
    clusters: np.ndarray
    first_centroids: np.ndarray
    potential: np.ndarray

    def __init__(self, k: int):
        self.k = k
        self.potential = np.empty(0)

    @abstractmethod
    def fit(self, X: np.ndarray):
        pass


class KMeans(Clustering):
    def fit(self, X: np.ndarray):
        self.d = X.shape[1]
        initial_centroids_index = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[initial_centroids_index, :]#中心点
        self.first_centroids = X[initial_centroids_index, :]

        n = X.shape[0]#要素の個数
        num = 0
        ranges = 0
        self.clusters = np.zeros(n)
        u=1
        oldarray = 0
        cen_max_dis=0 #中心の距離の最大値
        while True:
            num = num +1
            new_clusters = np.zeros(n)
            potential = 0.0
            
            new_ranges =0
            for i, x in enumerate(X):
                distances = np.linalg.norm(x - self.centroids, axis=1)
                new_clusters[i] = np.argmin(distances)
                potential += float(np.min(distances))
                new_range = np.max(distances)
                
            new_ranges = np.append(new_ranges, new_range)

            self.potential = np.append(self.potential, potential)

            # if np.array_equal(new_ranges,ranges):#クラスタの最長距離が変わらなかった時を終了条件にしたとき
            #     break
            # ranges = new_ranges

            # if np.array_equal(self.clusters, new_clusters):#終了判定
            #     break
            if num == eta :
                print("kmeans is not finish")
                break
            
            self.clusters = new_clusters

            new_centroids_distance=[]
            for i in range(self.k):
              for k in range(self.k):
                if i<k:
                  distance = np.linalg.norm(self.centroids[i]-self.centroids[k])
                  new_centroids_distance = np.append(new_centroids_distance,distance)
            if np.max(new_centroids_distance)==cen_max_dis: #中心同士の距離の最大値が変わらなかった時
              break
            cen_max_dis=np.max(new_centroids_distance)



            diss=0
            array=0
            
            for i in range(self.k):
                cluster = X[self.clusters == i, :] #それぞれの要素
                # for l in range(self.k):
                #   for m in range(self.k):
                #     distances = np.linalg.norm(cluster[l] - cluster[m])
                #     diss=np.append(diss,distances)
                # maxdiss=np.max(diss)
                # array=np.append(array,maxdiss) 
                    


                
                if cluster.shape[0] > 0: #追加
                    self.centroids[i] = np.mean(cluster, axis=0) #追加
          
            
              
            # if np.array_equal(oldarray,array):#クラスタの最長距離が変わらなかった時を終了条件にしたとき
            #     break
            oldarray=array
            
            

            


class KMeansPP(Clustering):
    def distance(self, x: np.ndarray):
        distances = np.linalg.norm(x - self.centroids, axis=1)
        return np.min(distances)

    def fit(self, X: np.ndarray):
        self.d = X.shape[1]

        first_centroid_index = np.random.choice(X.shape[0], 1)
        self.centroids = X[first_centroid_index, :]
        for _ in range(self.k - 1):
            distances = np.array([self.distance(x) for x in X])
            probabilities = distances**2 / np.sum(distances**2)
            new_centroid_index = np.random.choice(X.shape[0], 1, p=probabilities)
            self.centroids = np.append(self.centroids, X[new_centroid_index, :], axis=0)
            self.first_centroids = np.append(
                self.centroids, X[new_centroid_index, :], axis=0
            )
        num = 0
        n = X.shape[0]
        ranges = 0
        self.clusters = np.zeros(n)
        oldarray=0
        cen_max_dis=0
        cen_min_dis=0
        centroids_mean=[0,0]
        u=1

        while True:
            num = num +1
            new_clusters = np.zeros(n)
            potential = 0.0
            new_ranges=0
            new_centroids_mean=0
            for i, x in enumerate(X):
                distances = np.linalg.norm(x - self.centroids, axis=1)
                new_clusters[i] = np.argmin(distances)
                potential += float(np.min(distances))
                new_range = np.max(distances)
            new_ranges = np.append(new_ranges, new_range)
            self.potential = np.append(self.potential, potential)

            new_centroids_mean=np.mean(self.centroids,axis=0) #各重心の重心
            if np.array_equal(new_centroids_mean,centroids_mean):
              print("重心の重心の一致")
              print(num)
              break
            centroids_mean = new_centroids_mean

            

            new_centroids_distance=[]
            for i in range(self.k):
              for k in range(self.k):
                if i<k:
                  distance = np.linalg.norm(self.centroids[i]-self.centroids[k])
                  new_centroids_distance = np.append(new_centroids_distance,distance)
            
            if np.array_equal(self.clusters, new_clusters): #クラスタの分類が変わらなかった時　2番目に緩い条件
                if np.array_equal(new_ranges,ranges):#クラスタの最長距離が変わらなかった時を終了条件にしたとき
                  print("クラスタ内の最長距離が変わらない時も同時です")
                
                if np.max(new_centroids_distance)==cen_max_dis: 
                  print("中心の距離の最大値を距離関数とした時も同時です。")
                if np.min(new_centroids_distance)==cen_min_dis: 
                  print("中心の距離の最小値を距離関数とした時も同時です。")
                if np.array_equal(new_centroids_mean,centroids_mean):
                  print("各々の重心の重心を距離関数とした時も同時です")
                print("基本終了")
                break
            self.clusters = new_clusters

            if np.array_equal(new_ranges,ranges):#クラスタの最長距離が変わらなかった時を終了条件にしたとき　これが一番緩い条件?
                print("クラスタ内の中心との最長距離が変わらないので終了")
                break
            ranges = new_ranges

            if np.max(new_centroids_distance)==cen_max_dis: #中心同士の距離の最大値が変わらなかった時　一番厳しい条件?
              print("中心の距離の最大値を距離関数とした時の比較回数は")
              print(num)
              break
            if np.min(new_centroids_distance)==cen_min_dis: #中心同士の距離の最小値が変わらなかった時　一番厳しい条件?
              print("中心の距離の最小値を距離関数とした時の比較回数は")
              print(num)
              break
            cen_max_dis=np.max(new_centroids_distance)
            cen_min_dis=np.min(new_centroids_distance)

            if num == eta :
                print("kmeans++ is not finish")
                break

            self.clusters = new_clusters
            
            array=0
            diss =0
            for i in range(self.k):
                cluster = X[self.clusters == i, :]
                for l in range(self.k):
                  for m in range(self.k):
                    distances = np.linalg.norm(cluster[l] - cluster[m])
                    diss=np.append(diss,distances)
                maxdiss=np.max(diss)
                array=np.append(array,maxdiss)  

                if cluster.shape[0] > 0:
                    self.centroids[i] = np.mean(cluster, axis=0)
            if np.array_equal(oldarray,array):#クラスタの最長距離が変わらなかった時を終了条件にしたとき
                print("クラスタ内の最長距離が変わりませんでした。")
                break
            oldarray=array

            # 収束条件を色々変えてみて、どれも一様に収束すると思っていたが、実際に動かしてみると初期に応じて、距離関数を変えるとそれぞれ収束スピードが違う
            #クラスタ内の最長距離が最速で終了する時は誤った結果になりやすい(?)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    generator = np.random.default_rng(0)
    X = np.concatenate(
        [
            generator.normal(loc=(0, 0), scale=0.8, size=(100, 2)),
            generator.normal(loc=(4, 2), scale=0.8, size=(100, 2)),
            generator.normal(loc=(2, 5), scale=0.8, size=(100, 2)),
            generator.normal(loc=(5, 7), scale=0.8, size=(100, 2)),
            generator.normal(loc=(7, 4), scale=0.8, size=(100, 2)),
        ]
    )

    # kmeans = KMeans(5)
    kmeans_pp = KMeansPP(5)
    # kmeans.fit(X)
    kmeans_pp.fit(X)

    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    # ax[0][0].scatter(X[:, 0], X[:, 1], c=kmeans.clusters)
    # ax[0][0].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c="red")
    # ax[0][0].scatter(
    #     kmeans.first_centroids[:, 0], kmeans.first_centroids[:, 1], c="black" )
    # ax[0][0].set_title("KMeans")
    ax[0][1].scatter(X[:, 0], X[:, 1], c=kmeans_pp.clusters)
    ax[0][1].scatter(kmeans_pp.centroids[:, 0], kmeans_pp.centroids[:, 1], c="red")
    ax[0][1].scatter(
        kmeans_pp.first_centroids[:, 0], kmeans_pp.first_centroids[:, 1], c="black"
    )
    ax[0][1].set_title("KMeans++")
    # ax[1][0].plot(kmeans.potential)
    ax[1][1].plot(kmeans_pp.potential)
    plt.show()

    # print(kmeans.centroids)
    print(kmeans_pp.centroids)
    # print(kmeans.potential)
    print(kmeans_pp.potential)
