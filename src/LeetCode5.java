import java.util.Arrays;
import java.util.Scanner;

public class LeetCode5 {

	public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {//注意while处理多个case
            int m = in.nextInt();
            int n = in.nextInt();
            int A[] = new int[n];
            int i,j,k,count=0;
            for(i=0;i<n;i++)
                A[i] = in.nextInt();
            Arrays.sort(A);
            if(A[0]!=1){
                System.out.println(-1);
                continue;
            }
            int min[] = new int[m+1];
            int max;
            min[1] = 1;
            for(i=1;i<=m;i++) {
            	max = i;
            	for(j=0;j<n;j++) {
            		if(i-A[j]>=1 && i-A[j]<i) {
            			if(min[i-A[j]]+1<max)
            				max = min[i-A[j]]+1;
            		}else break;
            	}
            	min[i] = max;
            }
            System.out.println(min[m]);
        }
    }
	

}
