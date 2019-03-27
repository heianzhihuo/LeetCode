import java.util.Scanner;

public class LeetCode4 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {//注意while处理多个case
            int q = in.nextInt();
            
            for(int i=0;i<q;i++) {
            	int m = in.nextInt();
            	int n = in.nextInt();
            	if(m%2==0 && n%2==0) {
            		System.out.println(-(n-m)/2+m);
            	}else if(m%2==1 && n%2==1) {
            		System.out.println((n-m)%2+m);
            	}else if(m%2==0 && n%2==1) {
            		System.out.println(-(n-m+1)%2);
            	}else {
            		System.out.println((n-m+1)%2);
            	}
            }
           
        }

}
}
