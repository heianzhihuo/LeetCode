import java.util.HashSet;
import java.util.Scanner;
public class LeetCode6 {
	
	
	
	public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {//注意while处理多个case
            int n = in.nextInt();
            int m = in.nextInt();
            int res = Integer.MAX_VALUE;
            int seque[] = new int[n];
            for(int i=0;i<n;i++)
            	seque[i] = in.nextInt();
            for(int i=0;i<n;i++) {
            	HashSet<Integer> test = new HashSet<>();
            	for(int j=i;j<n;j++) {
            		if(seque[j]!=0)
            			test.add(seque[j]);
            		if(test.size()==m && j-i<res) {
            			res = j-i+1;
            			break;
            		}
            			
            	}
            }
            if(res<Integer.MAX_VALUE)
            	System.out.println(res);
            else
            	System.out.println(-1);
        }
    }
}
