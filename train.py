# train.py (RunPod Serverless에 업로드됨)
def handler(event):
    import pandas as pd
    import io
    import base64

    # Base64 인코딩된 CSV 받기
    encoded_csv = event["input"].get("csv_base64")
    decoded = base64.b64decode(encoded_csv)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    # 간단한 확인용 로직
    print("Received dataframe shape:", df.shape)

    # 예시 모델 학습 (로지스틱 회귀 등)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(df.drop("label", axis=1), df["label"])

    return {"status": "success", "rows": df.shape[0]}
