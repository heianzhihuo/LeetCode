import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class ByteDance1 {

	 public static void main(String[] args) {
		 Scanner in = new Scanner(System.in);
	        List<List<Integer>> data = new ArrayList<>();
	        while(in.hasNextLine()) {
	            String str = in.nextLine();
	            String strs[] = str.split(" ");
	            List<Integer> tmp = new ArrayList<>();
	            for(String t:strs) {
	                tmp.add(Integer.parseInt(t));
	            }
	            data.add(tmp);
	        }
	        int m = data.size();
	        if(m==0)
	            System.out.println(0);
	        int n = data.get(0).size();
	        int i,j;
	        int A[][] = new int[data.size()][data.get(0).size()];
	        for(i=0;i<m;i++)
	            for(j=0;j<n;j++)
	                A[i][j] = data.get(i).get(j);
	        int count = 0;
	        while(true) {
	        	boolean flag = false;
	        	int B[][] = new int[m][n];
	        	for(i=0;i<m;i++) 
	        		for(j=0;j<n;j++) 
	        			if(A[i][j]==1) {
	        				boolean f1 = false;
	        				if(i>0 && A[i-1][j]==2)
	        					f1 = true;
	        				if(j>0 && A[i][j-1]==2)
	        					f1 = true;
	        				if(i<m-1 && A[i+1][j]==2)
	        					f1 = true;
	        				if(j<n-1 && A[i][j+1]==2)
	        					f1 = true;
	        				if(f1) {
	        					B[i][j] = 2;
	        					flag = true;
	        				}else
	        					B[i][j] = A[i][j];
	        			}
	        			else B[i][j] = A[i][j];
	        	if(flag!=true)
	        		break;
	        	A = B;
	        	count++;
	        }
	        boolean fg = false;
	        for(i=0;i<m;i++) 
        		for(j=0;j<n;j++) 
        			if(A[i][j]==1) {
        				fg = true;
        				break;
        			}
        	if(fg)		
        		System.out.println(-1);
        	else System.out.println(count);
	        
	        
	    }

	 

}
