class Retrivers:
    def __init__(self,vectorstore):
        self.vectorstore=vectorstore
    
    def retrive(self,query:str,k=3,threshold=0.2):
        results=self.vectorstore.similarity_search_with_score(query, k=k)
        final_result=[]
        
        for doc, score in results:
            similarity_score=1-score
            if similarity_score>= threshold:
                final_result.append((doc,similarity_score))


        return final_result
    