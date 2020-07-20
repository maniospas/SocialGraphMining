package eu.h2020.helios_social.modules.socialgraphmining.GNN.operations;

public class Sort { 
	public static int[] sortedIndexes(double A[]) {
		int[] indexes = new int[A.length];
		for(int i=0;i<A.length;i++)
			indexes[i] = i;
		quick_sort(A, indexes, 0, A.length-1);
		return indexes;
	}
	
    private static int partition(double A[], int indexes[], int low, int high) { 
        double pi = A[high];  
        int i = (low-1); // smaller element index   
        for (int j=low; j<high; j++) { 
            if (A[j] <= pi) { 
                i++; 
                double tempA = A[i]; 
                A[i] = A[j]; 
                A[j] = tempA; 
                int tempI = indexes[i];
                indexes[i] = indexes[j]; 
                indexes[j] = tempI; 
            } 
        }
        double tempA = A[i+1]; 
        A[i+1] = A[high]; 
        A[high] = tempA; 
        int tempI = indexes[i+1]; 
        indexes[i+1] = indexes[high]; 
        indexes[high] = tempI; 
        return i+1; 
    } 
 
    private static void quick_sort(double A[], int indexes[], int low, int high) { 
        if (low < high) { 
            int pi = partition(A, indexes, low, high); 
   
            quick_sort(A, indexes, low, pi-1); 
            quick_sort(A, indexes, pi+1, high); 
        } 
    } 
}
