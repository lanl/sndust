program testread
implicit none

integer, parameter :: N = 4000
integer :: nc, io, nt, i
double precision :: t, xmcore, rb, lc
double precision, dimension(0:N) :: x,v
double precision, dimension(N) :: q,dq,u,deltam,abar,rho,temp,tempe,ye,pr,u2,uemit,xc,vc
character(len=*), parameter :: FMTH = "(3(A8, X) 6(A15, X))"
character(len=*), parameter :: FMTW = "(3(I8, X) 6(ES15.7, X))"
double precision, parameter :: urad = 1.0E9, utim = 1.0E1, utmp = 1.0E9, umas = 1.989E33
double precision, parameter :: uvel = urad / utim, uden = umas / (urad**3)

  nt = 0
  open(61, file='2out.bin', form='unformatted')
  open(62, file='mdl2.dat')

  write(62, FMTH) "nc", "nt", "id", "time", "xc", "vc", "mass", "rho", "temperature"
  do
    read(61, iostat=io) nc,t,xmcore,rb,lc,&
      (x(i),i=0,nc),(v(i),i=0,nc),(q(i),i=1,nc),(dq(i),i=1,nc),&
      (u(i),i=1,nc),(deltam(i),i=1,nc),(abar(i),i=1,nc),&
      (rho(i),i=1,nc),(temp(i),i=1,nc),(tempe(i),i=1,nc),&
      (ye(i),i=1,nc),&
      (pr(i),i=1,nc),(u2(i),i=1,nc),(uemit(i),i=1,nc)

    if(io > 0) then
      write(*,*) 'something wrong'
      stop
    else if (io < 0) then
      write(*,*) 'EoF'
      exit
    else
      nt = nt + 1
      xc(1:nc) = 0.5 * x(1:nc) + x(0:nc-1)
      vc(1:nc) = 0.5 * v(1:nc) + v(0:nc-1)
      do i=1, nc
        write(62, FMTW) nc, nt, i-1, t * utim, xc(i) * urad, vc(i) * uvel, deltam(i) * umas, rho(i) * uden, temp(i) * utmp
      enddo
    endif
  enddo

  close(61)
  close(62)
end program testread
