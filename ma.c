#include <stdio.h>

void showf(float f) {
    unsigned d = * (unsigned*) ((void*)&f);

    int k = 31;
    printf("%d", 1 & (d >> k--));
    printf(" ");

    for(int i=0; i<8; i++) printf("%d", 1 & (d >> k--));
    printf(" ");

    for(int i=0; i<7; i++) printf("%d", 1 & (d >> k--));
    printf("_");

    for(int i=0; i<8; i++) printf("%d", 1 & (d >> k--));
    printf("_");

    for(int i=0; i<8; i++) printf("%d", 1 & (d >> k--));
    printf("\n");


    int sign = 1 & (d >> 31);
    int exponent = (0xFF & (d >> 23)) - 0x7F;;
    printf("%c e%+d ", sign?'-':'+', exponent);

    int mantissa = 0x7FF & (d >> 0);
    printf("1.");
    for(int k=22; k>=0 && (d<<(31-k)); k--) printf("%d", 1 & (d >> k));
    printf("\n");


    if(sign) printf("-");

    if(exponent < 0) {
        printf(".");
        for(int e=exponent; e<-1; e++) printf("0");
        printf("1");
        for(int k=22; k>=0 && (d<<(31-k)); k--) printf("%d", 1 & (d >> k));
    }

    if(exponent == 0) {
        printf("1.");
        for(int k=22; k>=0 && (d<<(31-k)); k--) printf("%d", 1 & (d >> k));
    }

    if(exponent > 0) {
        printf("1");

        int k=22;
        for(int e=exponent; e>0; e--, k--) {
            printf("%d", k<0? 0: (1 & (d >> k)));
        }

        printf(".");

        for( ; k>=0 && (d<<(31-k)); k--)
            printf("%d", 1 & (d >> k));
    }

    printf("\n");

}

int main() {
    //float f = ((float) 0xAA) / ((float) 0x100);
    float f = (float) 0xF6F7 / (float) (1 << 16);
    showf(f);

    showf(f * (1<<15));
}
