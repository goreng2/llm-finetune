import json
from glob import glob


def main():
    result = []
    tmp = {}
    
    # JSON 파일 경로 파싱
    for i in glob("dataset/**/*.json"):
        print(i)
        
        # JSON 파일 내용 파싱
        with open(i, "r", encoding="cp949") as f:
            whole_data = json.load(f)
        
        # JSON 파일 Data 부분 파싱
        for idx, data in enumerate(whole_data["data"]):
            # print()
            # print(data["utterance"])
            # print(data["query"])
            # print()
            tmp.update(nl=data["utterance"], sql=data["query"])
            result.append(tmp)
            tmp = {}
            
        #     if idx == 3:
        #         break
            
        # break
    
    # Multiple json object 저장
    with open("dataset/nl2sql.json", "w", encoding="utf-8") as f:
        for data in result:
            f.write(json.dumps(data, ensure_ascii=False))
            f.write("\n")


if __name__ == "__main__":
    main()
