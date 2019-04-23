import java.util.Arrays;
import java.util.Scanner;

public class ByteDance {

	public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {// 注意，如果输入是多个测试用例，请通过while循环处理多个测试用例
            int k = in.nextInt();
            int d;
            for(d=0;d<k;d++) {
            	int n = in.nextInt();
                int a[] = new int[n];
                int i;
                for(i=0;i<n;i++)
                    a[i] = in.nextInt();
                Arrays.sort(a);
                if(n==2){
                    System.out.println(a[1]);
                }else if(n>2){
                    int sum = 0;
                    for(i=2;i<n;i++){
                        sum += a[i];
                        sum += a[1];
                    }
                    sum -= a[1];
                    System.out.println(sum);
                }
            }
        }
    }

}
