import java.util.Scanner;

public class LeetCode5 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {//注意while处理多个case
            int n = in.nextInt();
            int s = in.nextInt();
            for(int i=0;i<n;i++){
                int x = in.nextInt();
            }
            int sum = 1;
            int t = n-s;
            int m = 1000000007;
            for(int j=1;j<s;j++){
                sum *= (n-j+1)/j;
                sum  = sum % m;
            }
            for(int j=0;j<t;j++){
                sum *= 2;
                sum = sum % m;
            }
            System.out.println(sum);
        }
	}

}
