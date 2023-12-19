from langchain_helper import run_semantic_search

def main():
    try:
        results = run_semantic_search()

        # Handle or display the results as needed
        for result, score in results:
            print(f"Product: {result.page_content}, Similarity Score: {score}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
