import csv

positives = ["とても面白いです","簡単でわかりやすい","買ってよかったです"]
negatives = ["つまらないです","難しくて挫折しました","お金の無駄でした"]

dataset = [(text, 1)for text in positives] + [(text, 0)for text in negatives]

with open("data/sentiment_dataset.csv", "w", encoding="utf-8" , newline="") as f:
          writer = csv.writer(f)
          writer.writerow(["text","label"])
          writer.writerows(dataset)

