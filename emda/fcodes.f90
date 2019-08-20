subroutine resolution_grid(uc,mode,maxbin,nx,ny,nz,nbin,res_arr,bin_idx)
  implicit none
  real*8,    parameter :: PI = 3.14159
  integer,                intent(in) :: mode, maxbin,nx,ny,nz
  real,      dimension(6),intent(in) :: uc
  integer,   dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(out) :: bin_idx
  real,      dimension(0:maxbin-1),intent(out) :: res_arr
  integer,                         intent(out) :: nbin
  ! locals
  integer,   dimension(3)          :: nxyz
  real       :: low_res,high_res,resol,tmp_val,tmp_min,val,start,finish
  real       :: r(3),s1(3),step(3)
  integer    :: i,j,k,n,xyzmin(3),xyzmax(3),hkl(3),sloc,ibin,mnloc
  logical    :: debug
  !
  debug         = .FALSE.
  if(mode == 1) debug = .TRUE.
  call cpu_time(start)

  bin_idx = -100
  n = 0
  r = 0.0; s1 = 0.0
  res_arr = 0.0
  step = 0.0
  mnloc = -100
  xyzmin = 0; xyzmax = 0; hkl = 0

  nxyz = (/ nx, ny, nz /)

  xyzmin(1) = int(-nxyz(1)/2)
  xyzmin(2) = int(-nxyz(2)/2)
  xyzmin(3) = int(-nxyz(3)/2)
  xyzmax    = -(xyzmin+1)
  if(debug) print*, 'xyzmin = ', xyzmin
  if(debug) print*, 'xyzmax = ', xyzmax
  if(debug) print*, 'unit cell = ', uc
  call get_resol(uc,real(xyzmax(1)),0.0,0.0,r(1))
  call get_resol(uc,0.0,real(xyzmax(2)),0.0,r(2))
  call get_resol(uc,0.0,0.0,real(xyzmax(3)),r(3))
  if(debug) print*,'a-max, b-max, c-max = ', r
  !
  sloc = minloc(r,1)
  hkl = 0
  do i = 1, 3
     if(sloc == i) hkl(i) = sloc/sloc
  end do

  do i = 0, xyzmax(sloc)-1
     step = (i + 1.5) * hkl
     call get_resol(uc,step(1),step(2),step(3),resol)
     if(debug) print*, i,step(1),step(2),step(3),resol
     res_arr(i) = resol
     nbin = i + 1
  end do
  print*, 'nbin=', nbin
  high_res = res_arr(nbin-1)
  call get_resol(uc,0.0,0.0,0.0,low_res)
  print*,"Low res=",low_res,"High res=",high_res ,'A'

  print*, 'Creating resolution grid. Please wait...'

  ! Not using Friedel's Law
  do i=xyzmin(1), xyzmax(1)
     do j=xyzmin(2), xyzmax(2)
        do k=xyzmin(3), xyzmax(3)
           call get_resol(uc,real(i),real(j),real(k),resol)
           if(resol < high_res .or. resol > low_res) cycle
           n = n + 1
           ! Find the matching bin to resol
           do ibin = 0, nbin - 1
              val = sqrt((res_arr(ibin) - resol)**2)
              if(ibin == 0)then
                 tmp_val = val; tmp_min = val
                 mnloc = ibin 
              else
                 tmp_val = val
                 if(tmp_val < tmp_min)then
                    tmp_min = val
                    mnloc = ibin
                 end if
              end if
           end do
           bin_idx(i,j,k) = mnloc
        end do
     end do
  end do

  call cpu_time(finish)
  if(debug) print*, 'time for calculation(s) = ', finish-start
end subroutine resolution_grid

subroutine calc_fsc_using_halfmaps(hf1,hf2,bin_idx,nbin,mode,nx,ny,nz,Fo,Eo,bin_noise_var,bin_sgnl_var,bin_total_var,bin_fsc)
  implicit none
  real*8,    parameter :: PI = 3.14159

  integer,                intent(in) :: nbin,mode,nx,ny,nz
  complex*8, dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(in)  :: hf1,hf2
  complex*8, dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(out) :: Fo,Eo
  integer,   dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(in) :: bin_idx

  real*8,    dimension(0:nbin-1),intent(out) :: bin_sgnl_var,bin_noise_var,bin_total_var,bin_fsc
  !integer,                         intent(out) :: nbin
  ! locals
  integer,   dimension(3)          :: nxyz
  integer,   dimension(0:nbin-1) :: bin_arr_count

  real*8,    dimension(0:nbin-1) :: A1_sum,B1_sum,A2_sum,B2_sum,A1A2_sum,B1B2_sum
  real*8,    dimension(0:nbin-1) :: A1A1_sum,B1B1_sum,A2A2_sum,B2B2_sum
  real*8,    dimension(0:nbin-1) :: bin_arr_fdiff,A_sum,B_sum,AA_sum,BB_sum
  real*8,    dimension(0:nbin-1) :: F1_var, F2_var, F1F2_covar
  !
  complex*8  :: fdiff
  real*8     :: A,B,A1,A2,B1,B2,bin_sigvar
  real       :: start,finish
  integer    :: i,j,k,xyzmin(3),xyzmax(3),ibin
  logical    :: debug,make_all_zero
  !
  debug         = .FALSE.
  make_all_zero = .FALSE.
  if(mode == 1) debug = .TRUE.
  call cpu_time(start)

  Fo = 0.0
  Eo = 0.0

  bin_arr_fdiff = 0.0
  bin_sigvar    = 0.0
  bin_noise_var = 0.0
  bin_sgnl_var  = 0.0
  bin_total_var = 0.0

  F1F2_covar = 0.0
  F1_var = 0.0
  F2_var = 0.0
  bin_total_var = 0.0
  bin_fsc = 0.0

  A_sum = 0.0
  B_sum = 0.0
  AA_sum = 0.0
  BB_sum = 0.0

  A1_sum = 0.0; A2_sum = 0.0
  B1_sum = 0.0; B2_sum = 0.0
  A1A2_sum = 0.0; B1B2_sum = 0.0
  A1A1_sum = 0.0; B1B1_sum = 0.0
  A2A2_sum = 0.0; B2B2_sum = 0.0

  bin_arr_count = 0
  xyzmin = 0; xyzmax = 0
  nxyz = (/ nx, ny, nz /)

  xyzmin(1) = int(-nxyz(1)/2)
  xyzmin(2) = int(-nxyz(2)/2)
  xyzmin(3) = int(-nxyz(3)/2)
  xyzmax    = -(xyzmin+1)
  if(debug) print*, 'xyzmin = ', xyzmin
  if(debug) print*, 'xyzmax = ', xyzmax
  
  

  print*, 'bin_arr_count=', sum(bin_arr_count)

  ! Not using Friedel's Law
  do i=xyzmin(1), xyzmax(1)
     do j=xyzmin(2), xyzmax(2)
        do k=xyzmin(3), xyzmax(3)
           if(bin_idx(i,j,k) < 0 .or. bin_idx(i,j,k) > nbin-1) cycle
           bin_arr_count(bin_idx(i,j,k)) = bin_arr_count(bin_idx(i,j,k)) + 1
           !New calculation of total-var and noise var
           fdiff = hf1(i,j,k) - hf2(i,j,k)
           bin_arr_fdiff(bin_idx(i,j,k)) = bin_arr_fdiff(bin_idx(i,j,k)) + real(fdiff * conjg(fdiff))
           Fo(i,j,k) = (hf1(i,j,k) + hf2(i,j,k))/2.0
           A = real(Fo(i,j,k)); B = aimag(Fo(i,j,k))
           A_sum(bin_idx(i,j,k)) = A_sum(bin_idx(i,j,k)) + A
           AA_sum(bin_idx(i,j,k)) = AA_sum(bin_idx(i,j,k)) + A*A
           B_sum(bin_idx(i,j,k)) = B_sum(bin_idx(i,j,k)) + B
           BB_sum(bin_idx(i,j,k)) = BB_sum(bin_idx(i,j,k)) + B*B
           ! end of new calculation

           ! correspondence hf1 : A1 + iB1 ; hf2 = A2 + iB2
           A1 = real(hf1(i,j,k));  A2 = real(hf2(i,j,k))
           B1 = aimag(hf1(i,j,k)); B2 = aimag(hf2(i,j,k))
           A1_sum(bin_idx(i,j,k)) = A1_sum(bin_idx(i,j,k)) + A1
           A2_sum(bin_idx(i,j,k)) = A2_sum(bin_idx(i,j,k)) + A2
           B1_sum(bin_idx(i,j,k)) = B1_sum(bin_idx(i,j,k)) + B1
           B2_sum(bin_idx(i,j,k)) = B2_sum(bin_idx(i,j,k)) + B2

           A1A2_sum(bin_idx(i,j,k)) = A1A2_sum(bin_idx(i,j,k)) + A1 * A2
           B1B2_sum(bin_idx(i,j,k)) = B1B2_sum(bin_idx(i,j,k)) + B1 * B2

           A1A1_sum(bin_idx(i,j,k)) = A1A1_sum(bin_idx(i,j,k)) + A1 * A1
           B1B1_sum(bin_idx(i,j,k)) = B1B1_sum(bin_idx(i,j,k)) + B1 * B1

           A2A2_sum(bin_idx(i,j,k)) = A2A2_sum(bin_idx(i,j,k)) + A2 * A2
           B2B2_sum(bin_idx(i,j,k)) = B2B2_sum(bin_idx(i,j,k)) + B2 * B2
        end do
     end do
  end do

  if(debug) print*, 'bin_arr_count=', sum(bin_arr_count)

  do ibin=0, nbin-1 !to make compatible with python arrays
     bin_noise_var(ibin) = bin_arr_fdiff(ibin) / (bin_arr_count(ibin) * 4)
     bin_total_var(ibin) = (AA_sum(ibin) + BB_sum(ibin))/bin_arr_count(ibin) &
          - ((A_sum(ibin)/bin_arr_count(ibin))**2 + (B_sum(ibin)/bin_arr_count(ibin))**2)
     
     F1F2_covar(ibin) = (A1A2_sum(ibin) + B1B2_sum(ibin)) / bin_arr_count(ibin) - &
          (A1_sum(ibin) / bin_arr_count(ibin) * A2_sum(ibin) / bin_arr_count(ibin) + &
          B1_sum(ibin) / bin_arr_count(ibin) * B2_sum(ibin) / bin_arr_count(ibin))

     F1_var(ibin) = (A1A1_sum(ibin) + B1B1_sum(ibin))/bin_arr_count(ibin) - &
          ((A1_sum(ibin)/bin_arr_count(ibin))**2 + (B1_sum(ibin)/bin_arr_count(ibin))**2)
     F2_var(ibin) = (A2A2_sum(ibin) + B2B2_sum(ibin))/bin_arr_count(ibin) - &
          ((A2_sum(ibin)/bin_arr_count(ibin))**2 + (B2_sum(ibin)/bin_arr_count(ibin))**2)

     bin_sgnl_var(ibin) = F1F2_covar(ibin)
     bin_fsc(ibin) = F1F2_covar(ibin) / (sqrt(F1_var(ibin)) * sqrt(F2_var(ibin)))   
     if(debug)then
        print*,ibin,bin_noise_var(ibin),bin_sgnl_var(ibin), &
             bin_total_var(ibin),bin_fsc(ibin),bin_arr_count(ibin)
     end if
  end do

  ! Calculate normalized structure factors
  do i=xyzmin(1), xyzmax(1)
     do j=xyzmin(2), xyzmax(2)
        do k=xyzmin(3), xyzmax(3)
           if(bin_idx(i,j,k) < 0 .or. bin_idx(i,j,k) > nbin-1) cycle
           if(bin_sgnl_var(bin_idx(i,j,k)) == 0.0) cycle
           Eo(i,j,k) = Fo(i,j,k)/sqrt(bin_total_var(bin_idx(i,j,k)))
        end do
     end do
  end do
  
  call cpu_time(finish)
  if(debug) print*, 'time for calculation(s) = ', finish-start
end subroutine calc_fsc_using_halfmaps

subroutine calc_covar_and_fsc_betwn_anytwomaps(hf1,hf2,bin_idx,nbin,mode,F1F2_covar,bin_fsc,nx,ny,nz)
  implicit none
  integer,intent(in) :: nbin,mode,nx,ny,nz
  integer,  dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2),intent(in)  :: bin_idx
  complex*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2),intent(in)  :: hf1,hf2
  real*8,   dimension(0:nbin-1), intent(out) :: F1F2_covar,bin_fsc
  !
  integer,  dimension(0:nbin-1) :: bin_arr_count
  real*8,   dimension(0:nbin-1) :: F1_var,F2_var,A1_sum,B1_sum,A2_sum,B2_sum,A1A2_sum,B1B2_sum
  real*8,   dimension(0:nbin-1) :: A1A1_sum,B1B1_sum,A2A2_sum,B2B2_sum
  real*8    :: A1,A2,B1,B2
  !real      :: bin_sigvar
  !complex*8 :: fdiff, Fo
  integer   :: i,j,k,xmin,xmax,ymin,ymax,zmin,zmax,ibin
  real      :: start, finish
  logical   :: debug, make_all_zero 
  !
  debug = .FALSE.
  make_all_zero = .FALSE.
  if(mode == 1) debug = .TRUE.
  call cpu_time(start)
 
  F1F2_covar = 0.0
  F1_var = 0.0
  F2_var = 0.0
  bin_fsc = 0.0

  A1_sum = 0.0; A2_sum = 0.0
  B1_sum = 0.0; B2_sum = 0.0
  A1A2_sum = 0.0; B1B2_sum = 0.0
  A1A1_sum = 0.0; B1B1_sum = 0.0
  A2A2_sum = 0.0; B2B2_sum = 0.0

  xmin = int(-nx/2); xmax = -(xmin+1)
  ymin = int(-ny/2); ymax = -(ymin+1)
  zmin = int(-nz/2); zmax = -(zmin+1)
  if(debug) print*, '[',xmin,xmax,'],[', ymin,ymax,'],[',zmin,zmax,']'

  bin_arr_count = 0

  do i=xmin, xmax
     do j=ymin, ymax
        do k=zmin, zmax
           if(bin_idx(i,j,k) < 0 .or. bin_idx(i,j,k) > nbin-1)then
              cycle
           else
              bin_arr_count(bin_idx(i,j,k)) = bin_arr_count(bin_idx(i,j,k)) + 1
              ! correspondence hf1 : A1 + iB1 ; hf2 = A2 + iB2
              A1 = real(hf1(i,j,k));  A2 = real(hf2(i,j,k))
              B1 = aimag(hf1(i,j,k)); B2 = aimag(hf2(i,j,k))
              A1_sum(bin_idx(i,j,k)) = A1_sum(bin_idx(i,j,k)) + A1
              A2_sum(bin_idx(i,j,k)) = A2_sum(bin_idx(i,j,k)) + A2
              B1_sum(bin_idx(i,j,k)) = B1_sum(bin_idx(i,j,k)) + B1
              B2_sum(bin_idx(i,j,k)) = B2_sum(bin_idx(i,j,k)) + B2

              A1A2_sum(bin_idx(i,j,k)) = A1A2_sum(bin_idx(i,j,k)) + A1 * A2
              B1B2_sum(bin_idx(i,j,k)) = B1B2_sum(bin_idx(i,j,k)) + B1 * B2

              A1A1_sum(bin_idx(i,j,k)) = A1A1_sum(bin_idx(i,j,k)) + A1 * A1
              B1B1_sum(bin_idx(i,j,k)) = B1B1_sum(bin_idx(i,j,k)) + B1 * B1

              A2A2_sum(bin_idx(i,j,k)) = A2A2_sum(bin_idx(i,j,k)) + A2 * A2
              B2B2_sum(bin_idx(i,j,k)) = B2B2_sum(bin_idx(i,j,k)) + B2 * B2
           end if
        end do
     end do
  end do
  if(debug) print*,'ibin F1F2_covar(ibin) F1_var(ibin) F2_var(ibin) bin_fsc(ibin) bin_reflex_count'
  do ibin=0, nbin-1 !to make compatible with python arrays

     F1F2_covar(ibin) = (A1A2_sum(ibin) + B1B2_sum(ibin)) / bin_arr_count(ibin) - &
          (A1_sum(ibin) / bin_arr_count(ibin) * A2_sum(ibin) / bin_arr_count(ibin) + &
          B1_sum(ibin) / bin_arr_count(ibin) * B2_sum(ibin) / bin_arr_count(ibin))

     F1_var(ibin) = (A1A1_sum(ibin) + B1B1_sum(ibin))/bin_arr_count(ibin) - &
          ((A1_sum(ibin)/bin_arr_count(ibin))**2 + (B1_sum(ibin)/bin_arr_count(ibin))**2)
     F2_var(ibin) = (A2A2_sum(ibin) + B2B2_sum(ibin))/bin_arr_count(ibin) - &
          ((A2_sum(ibin)/bin_arr_count(ibin))**2 + (B2_sum(ibin)/bin_arr_count(ibin))**2)
     
     !bin_sgnl_var(ibin) = F1F2_covar(ibin)
     bin_fsc(ibin) = F1F2_covar(ibin) / (sqrt(F1_var(ibin)) * sqrt(F2_var(ibin)))

     if(debug)then
        print*,ibin,F1F2_covar(ibin),F1_var(ibin),F2_var(ibin),bin_fsc(ibin),bin_arr_count(ibin)
     end if
  end do

  call cpu_time(finish)
  if(debug) print*, 'time for loop = ', finish-start
  return
end subroutine calc_covar_and_fsc_betwn_anytwomaps


subroutine read_into_grid(bin_idx,bin_fsc,nbin,nx,ny,nz,fsc_weighted_grid)
  implicit none
  integer,   intent(in) :: nbin,nx,ny,nz
  real,      dimension(0:nbin-1),intent(in) :: bin_fsc
  integer,   dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(in)  :: bin_idx
  real,      dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(out) :: fsc_weighted_grid
  ! locals
  integer,   dimension(3) :: nxyz
  integer    :: i,j,k,xyzmin(3),xyzmax(3)!,ibin
  !
  xyzmin = 0; xyzmax = 0
  fsc_weighted_grid = 0.0

  nxyz = (/ nx, ny, nz /)

  xyzmin(1) = int(-nxyz(1)/2)
  xyzmin(2) = int(-nxyz(2)/2)
  xyzmin(3) = int(-nxyz(3)/2)
  xyzmax    = -(xyzmin+1)
  
  do i=xyzmin(1), xyzmax(1)
     do j=xyzmin(2), xyzmax(2)
        do k=xyzmin(3), xyzmax(3)
           if(bin_idx(i,j,k) < 0 .or. bin_idx(i,j,k) > nbin-1) cycle
           fsc_weighted_grid(i,j,k) = bin_fsc(bin_idx(i,j,k))
        end do
     end do
  end do

end subroutine read_into_grid


subroutine get_resol(uc,h,k,l,resol)
  implicit none
  real,dimension(6),intent(in) :: uc
  !integer,intent(in) :: h,k,l
  real,intent(in) :: h,k,l
  real :: a,b,c,vol,sa,sb,sc,s2,tmp
  real,intent(out) :: resol
  !
  a = uc(1); b = uc(2); c = uc(3)
  !print*, a,b,c
  vol = a*b*c
  sa = b*c/vol; sb = a*c/vol; sc = a*b/vol
  s2 = ((h*sa)**2 + (k*sb)**2 + (l*sc)**2)/4.0
  if(s2 == 0.0) s2 = 1.0e-10 ! F(000) resolution hard coded
  tmp = sqrt(s2)
  resol = 1.0/(2.0*tmp)
  return
end subroutine get_resol

subroutine get_st(nx,ny,nz,t,st,s1,s2,s3)
  implicit none
  real*8, parameter :: PI = 3.14159
  integer,intent(in) :: nx,ny,nz
  real,dimension(3),intent(in) :: t
  real :: sv(3)
  complex :: xj
  complex*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2),intent(out) :: st
  integer,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2),intent(out) :: s1,s2,s3
  integer :: i,j,k,xmin,xmax,ymin,ymax,zmin,zmax

  xmin = int(-nx/2); xmax = -(xmin+1)
  ymin = int(-ny/2); ymax = -(ymin+1)
  zmin = int(-nz/2); zmax = -(zmin+1)

  xj = (0,1)

  do i=xmin, xmax
     do j=ymin, ymax
        do k=zmin, zmax
           s1(i,j,k) = i
           s2(i,j,k) = j
           s3(i,j,k) = k
           sv(1) = i
           sv(2) = j
           sv(3) = k
           st(i,j,k) = exp(2.0 * PI * xj * dot_product(t,sv))
        end do
     end do
  end do
  return
end subroutine get_st

subroutine fsc_weight_calculation(fsc_weighted_grid,bin_fsc,F1,F2,bin_idx,nbin,mode,nx,ny,nz)
  implicit none
  integer,intent(in) :: nbin,mode,nx,ny,nz
  complex*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2),intent(in) :: F1,F2
  integer,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2),intent(in) :: bin_idx 
  real*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2),intent(out) :: fsc_weighted_grid
  real*8,dimension(0:nbin-1),intent(out) :: bin_fsc
  real*8,dimension(0:nbin-1) :: F1F1_sum,F2F2_sum,F1F2_sum,fsc_weight
  integer,dimension(0:nbin-1) :: bin_arr_count
  integer :: i,j,k,n,xmin,xmax,ymin,ymax,zmin,zmax,ibin
  real :: start, finish
  logical :: debug
  !
  debug = .FALSE.
  if(mode == 1) debug = .TRUE.
  call cpu_time(start)
  xmin = int(-nx/2); xmax = -(xmin+1)
  ymin = int(-ny/2); ymax = -(ymin+1)
  zmin = int(-nz/2); zmax = -(zmin+1)
  if(debug) print*, '[',xmin,xmax,'],[', ymin,ymax,'],[',zmin,zmax,']'

  bin_arr_count = 0
  F1F1_sum = 0.0
  F2F2_sum = 0.0
  F1F2_sum = 0.0
  bin_fsc  = 0.0
  fsc_weight = 0.0
  fsc_weighted_grid = 0.0

  do i=xmin, xmax
     do j=ymin, ymax
        do k=zmin, zmax
           if(bin_idx(i,j,k) < 0 .or. bin_idx(i,j,k) > nbin-1)then
              n = n + 1
              cycle
           else
              bin_arr_count(bin_idx(i,j,k)) = bin_arr_count(bin_idx(i,j,k)) + 1
              F1F1_sum(bin_idx(i,j,k)) = F1F1_sum(bin_idx(i,j,k)) + real(F1(i,j,k) * conjg(F1(i,j,k)))
              F2F2_sum(bin_idx(i,j,k)) = F2F2_sum(bin_idx(i,j,k)) + real(F2(i,j,k) * conjg(F2(i,j,k)))
              F1F2_sum(bin_idx(i,j,k)) = F1F2_sum(bin_idx(i,j,k)) + real(F1(i,j,k) * conjg(F2(i,j,k)))
           end if
        end do
     end do
  end do
  if(debug) print*,'Number of reflex outside the range: ',n
  if(debug) print*,'ibin   bin_fsc(F1,F2)   bin_fsc_weight   bin_reflex_count'
  do ibin=0, nbin-1 !to make compatible with python arrays
     if(F1F1_sum(ibin) == 0.0 .or. F2F2_sum(ibin) == 0.0 )then
        bin_fsc(ibin) = 0.0
        fsc_weight(ibin)    = 0.0
     else
        bin_fsc(ibin)       = F1F2_sum(ibin) / (sqrt(F1F1_sum(ibin)) * sqrt(F2F2_sum(ibin)))
        fsc_weight(ibin)    = -1.0 * bin_fsc(ibin) / (1.0 - bin_fsc(ibin)**2)
     end if
     if(debug)then
        print*,ibin,bin_fsc(ibin),fsc_weight(ibin),bin_arr_count(ibin)
     end if
  end do

  do i=xmin, xmax
     do j=ymin, ymax
        do k=zmin, zmax
           if(bin_idx(i,j,k) < 0 .or. bin_idx(i,j,k) > nbin-1)then
              fsc_weighted_grid(i,j,k) = 0.0
              cycle
           else
              fsc_weighted_grid(i,j,k) = fsc_weight(bin_idx(i,j,k))
           end if
        end do
     end do
  end do

  call cpu_time(finish)
  if(debug) print*, 'time for fsc and weights calculation = ', finish-start
end subroutine fsc_weight_calculation

subroutine calc_avg_maps(all_maps,bin_idx,wgt,Bf_arr,uc,nbin,nmaps,nbf,mode,nx,ny,nz,avgmaps_all)
  implicit none
  !
  integer,  intent(in) :: nbin,nmaps,nbf,mode,nx,ny,nz
  real,     dimension(6),                                           intent(in) :: uc
  real,     dimension(nbf),                                         intent(in) :: Bf_arr
  real*8,   dimension(0:nmaps-1,0:nmaps-1,0:nbin-1),                intent(in) :: wgt
  !real*8,   dimension(0:nbin-1),                                    intent(in) :: res_arr
  integer,  dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2),intent(in) :: bin_idx 
  complex*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2,0:nmaps-1),intent(in) :: all_maps
  complex*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2,0:nmaps-1,nbf),intent(out) :: avgmaps_all
  integer :: i,j,k,xmin,xmax,ymin,ymax,zmin,zmax,imap,jmap,ibf
  real    :: resol,s,start, finish, lowres, highres
  real    :: Bfac(nbf)
  logical :: debug
  !
  debug = .FALSE.
  if(mode == 1) debug = .TRUE.
  call cpu_time(start)
  xmin = int(-nx/2); xmax = -(xmin+1)
  ymin = int(-ny/2); ymax = -(ymin+1)
  zmin = int(-nz/2); zmax = -(zmin+1)
  if(debug) print*, '[',xmin,xmax,'],[', ymin,ymax,'],[',zmin,zmax,']'

  avgmaps_all = 0.0
  Bfac = -1.0 * Bf_arr

  call get_resol(uc,0.0,0.0,0.0,lowres)
  call get_resol(uc,real(xmin),real(ymin),real(zmin),highres)
  print*,'low high resol= ', lowres, highres

  print*, 'Calculating average maps...'
  do i=xmin, xmax
     do j=ymin, ymax
        do k=zmin, zmax
           if(bin_idx(i,j,k) < 0 .or. bin_idx(i,j,k) > nbin-1)then
              !avgmaps_all(i,j,k) = 0.0
              cycle
           else
              !s = 1.0 / res_arr(bin_idx(i,j,k)) ! Assuming that resolution never becomes zero
              call get_resol(uc,real(i),real(j),real(k),resol)
              if(resol == 0.0)then
                 print*, i,j,k
              end if

              s = 1.0 / resol
              do imap=0, nmaps-1
                 do jmap=0, nmaps-1
                    do ibf = 1, nbf
                       avgmaps_all(i,j,k,imap,ibf) = avgmaps_all(i,j,k,imap,ibf) + &
                            all_maps(i,j,k,jmap) * wgt(imap,jmap,bin_idx(i,j,k)) * &
                            exp(-1.0 * (Bfac(ibf)/4.0) * s**2) !B-factor sharpening/blurring
                    end do
                 end do
              end do
           end if
        end do
     end do
  end do

  call cpu_time(finish)
  if(debug) print*, 'time for avgerage maps calculation = ', finish-start  
end subroutine calc_avg_maps

subroutine calc_avg_maps_3d(all_maps,bin_idx,Smat,Fmat,Tinv,Bf_arr,uc,nbin, &
     nmaps,nbf,mode,nx,ny,nz,avgmaps_all)
  implicit none
  !
  integer,  intent(in) :: nbin,nmaps,nbf,mode,nx,ny,nz
  real,     dimension(6),                                           intent(in) :: uc
  real,     dimension(nbf),                                         intent(in) :: Bf_arr
  real*8,   dimension(0:nmaps-1,0:nmaps-1,0:nbin-1),                intent(in) :: Smat,Tinv
  integer,  dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2),intent(in) :: bin_idx
  real,     dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2,0:nmaps-1,0:nmaps-1),intent(in) :: Fmat
  complex*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2,0:nmaps-1),intent(in) :: all_maps
  complex*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2,0:nmaps-1,nbf),intent(out) :: avgmaps_all
  ! locals
  real, dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2,0:nmaps-1,0:nmaps-1) :: F_dot_Tinv
  real, dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2,0:nmaps-1,0:nmaps-1) :: S_dot_FdotTinv
  integer :: i,j,k,xmin,xmax,ymin,ymax,zmin,zmax,imap,jmap,ibf
  real    :: resol,s,start, finish, lowres, highres
  real    :: Bfac(nbf)
  logical :: debug
  !
  debug = .FALSE.
  if(mode == 1) debug = .TRUE.
  call cpu_time(start)
  xmin = int(-nx/2); xmax = -(xmin+1)
  ymin = int(-ny/2); ymax = -(ymin+1)
  zmin = int(-nz/2); zmax = -(zmin+1)
  if(debug) print*, '[',xmin,xmax,'],[', ymin,ymax,'],[',zmin,zmax,']'

  avgmaps_all = 0.0
  Bfac = -1.0 * Bf_arr
  F_dot_Tinv = 0.0
  S_dot_FdotTinv = 0.0

  call get_resol(uc,0.0,0.0,0.0,lowres)
  call get_resol(uc,real(xmin),real(ymin),real(zmin),highres)
  print*,'low high resol= ', lowres, highres

  print*, 'Calculating average maps...'
  ! New code
  do i=xmin, xmax
     do j=ymin, ymax
        do k=zmin, zmax
           if(bin_idx(i,j,k) < 0 .or. bin_idx(i,j,k) > nbin-1)then
              !avgmaps_all(i,j,k) = 0.0
              cycle
           else
              call get_resol(uc,real(i),real(j),real(k),resol)
              if(resol == 0.0)then
                 print*, i,j,k
              end if
              !
              do imap=0, nmaps-1
                 do jmap=0, nmaps-1
                    F_dot_Tinv(i,j,k,imap,jmap) = F_dot_Tinv(i,j,k,imap,jmap) + &
                         Fmat(i,j,k,imap,jmap) * Tinv(jmap,imap,bin_idx(i,j,k))
                 end do
              end do
              do imap=0, nmaps-1
                 do jmap=0, nmaps-1
                    S_dot_FdotTinv(i,j,k,imap,jmap) = S_dot_FdotTinv(i,j,k,imap,jmap) + &
                         Smat(imap,jmap,bin_idx(i,j,k)) * F_dot_Tinv(i,j,k,jmap,imap) 
                 end do
              end do
              !
              s = 1.0 / resol
              do imap=0, nmaps-1
                 do jmap=0, nmaps-1
                    do ibf = 1, nbf
                       avgmaps_all(i,j,k,imap,ibf) = avgmaps_all(i,j,k,imap,ibf) + &
                            S_dot_FdotTinv(i,j,k,imap,jmap) * all_maps(i,j,k,jmap) * &
                            exp(-1.0 * (Bfac(ibf)/4.0) * s**2) !B-factor sharpening/blurring
                    end do
                 end do
              end do
           end if
        end do
     end do
  end do
  ! End new code

  call cpu_time(finish)
  if(debug) print*, 'time for fsc and weights calculation = ', finish-start  

end subroutine calc_avg_maps_3d

subroutine apply_bfactor_to_map(mapin,Bf_arr,uc,nx,ny,nz,nbf,mode,all_mapout)
  implicit none
  !
  integer,  intent(in) :: nbf,mode,nx,ny,nz
  real,     dimension(6),                                           intent(in) :: uc
  real,     dimension(nbf),                                         intent(in) :: Bf_arr
  complex*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2),intent(in) :: mapin
  complex*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2,nbf),intent(out) :: all_mapout
  integer :: i,j,k,xmin,xmax,ymin,ymax,zmin,zmax,ibf
  real    :: resol,s,start, finish!, lowres, highres
  real    :: Bfac(nbf)
  logical :: debug
  !
  debug = .FALSE.
  if(mode == 1) debug = .TRUE.
  call cpu_time(start)
  xmin = int(-nx/2); xmax = -(xmin+1)
  ymin = int(-ny/2); ymax = -(ymin+1)
  zmin = int(-nz/2); zmax = -(zmin+1)
  if(debug) print*, '[',xmin,xmax,'],[', ymin,ymax,'],[',zmin,zmax,']'

  all_mapout = 0.0
  Bfac = -1.0 * Bf_arr

  !call get_resol(uc,0.0,0.0,0.0,lowres)
  !call get_resol(uc,real(xmin),real(ymin),real(zmin),highres)
  !print*,'low high resol= ', lowres, highres

  print*, 'Applying B factors to map...'
  do i=xmin, xmax
     do j=ymin, ymax
        do k=zmin, zmax
           call get_resol(uc,real(i),real(j),real(k),resol)
           if(resol == 0.0)then
              print*, i,j,k
           end if

           s = 1.0 / resol
           do ibf = 1, nbf
              all_mapout(i,j,k,ibf) = all_mapout(i,j,k,ibf) + &
                   mapin(i,j,k) * exp(-1.0 * (Bfac(ibf)/4.0) * s**2) !B-factor sharpening/blurring
           end do
        end do
     end do
  end do
  call cpu_time(finish)
  if(debug) print*, 'time for map blurring/sharpening = ', finish-start 

end subroutine apply_bfactor_to_map


subroutine tricubic(RM,F,FRS,ncopies,mode,nx,ny,nz)
  implicit none
  real*8,dimension(3,3),intent(in):: RM
  integer,intent(in):: nx,ny,nz,mode,ncopies
  complex*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2,ncopies),intent(in):: F
  complex*8,dimension(-nx/2:(nx-2)/2,-ny/2:(ny-2)/2,-nz/2:(nz-2)/2,ncopies),intent(out):: FRS
  integer :: x1(3,-1:2)
  real*8 :: x(3),xd(3),s(3)
  real*8 :: ul,ul2
  real*8 :: vl(4)
  !real :: high_res,resol
  integer :: i,j,h,k,l,nmin
  logical :: debug
  integer :: nxyz(3),nxyzmn(3),nxyzmx(3)
  integer :: xmin,xmax,ymin,ymax,zmin,zmax,ic
  complex*8 :: fl(-1:2,-1:2,-1:2,ncopies),esz(-1:2,-1:2,ncopies),esyz(-1:2,ncopies)
  complex*8 :: esxyz(ncopies)
  !
  debug = .FALSE.
  if(mode == 1) debug = .TRUE.
  !   Body
  nxyz(1) = nx; nxyz(2) = ny; nxyz(3) =nz
  nxyzmn(1) = -nx/2; nxyzmn(2) = -ny/2; nxyzmn(3) = -nz/2
  nxyzmx(1) = (nx-2)/2; nxyzmx(2) = (ny-2)/2; nxyzmx(3) = (nz-2)/2
  nmin = min(nx,ny,nz)

  xmin = int(-nx/2); xmax = -(xmin+1)
  ymin = int(-ny/2); ymax = -(ymin+1)
  zmin = int(-nz/2); zmax = -(zmin+1)

  if(debug) write(*,*) nxyz,nxyzmn,nxyzmx

  do l = zmin, zmax
     do k = ymin, ymax
        do h = xmin, xmax
           s(1) = h
           s(2) = k
           s(3) = l
           x = matmul(transpose(RM),s)
           do i = 1, 3
              x1(i,0) = floor(x(i))
              xd(i) = x(i) - real(x1(i,0))
              if(abs(xd(i)).gt.1.0) then
                 print*, 'Something is wrong ',xd(i)
                 stop
              endif
              x1(i,1)  = x1(i,0) + 1
              x1(i,2)  = x1(i,0) + 2
              x1(i,-1) = x1(i,0) - 1
           end do
           !
           !  Careful here: we may get to the outside of the array
           do i = 1,3
              do j= -1,2
                 x1(i,j) = min(nxyzmx(i),max(nxyzmn(i),x1(i,j)))
              enddo
           enddo
           do ic=1, ncopies
              fl(-1:2,-1:2,-1:2,ic) = F(x1(1,-1:2),x1(2,-1:2),x1(3,-1:2),ic)
           end do

           !
           !  Alternattive implementation
           !  along z
           ul = xd(3)
           ul2 = ul*ul
           vl(1) = ul*((2.0d0-ul)*ul-1.0d0)
           vl(2) = ul2*(3.0d0*ul-5.0d0)+2.0d0
           vl(3) = ul*((4.0d0-3.0d0*ul)*ul+1.0d0)
           vl(4) = ul2*(ul-1.0d0)
           vl = 0.5d0*vl
           do ic=1,ncopies
              do j=-1,2
                 do i=-1,2
                    esz(i,j,ic) = dot_product(vl,fl(i,j,-1:2,ic))
                 enddo
              enddo
           end do
           ul = xd(2)
           ul2 = ul*ul
           vl(1) = ul*((2.0d0-ul)*ul-1.0d0)
           vl(2) = ul2*(3.0d0*ul-5.0d0)+2.0d0
           vl(3) = ul*((4.0d0-3.0d0*ul)*ul+1.0d0)
           vl(4) = ul2*(ul-1.0d0)
           vl = 0.5d0*vl
           do ic=1,ncopies
              do i=-1,2
                 esyz(i,ic) = dot_product(vl,esz(i,-1:2,ic))
              enddo
           end do
           ul = xd(1)
           ul2 = ul*ul
           vl(1) = ul*((2.0d0-ul)*ul-1.0d0)
           vl(2) = ul2*(3.0d0*ul-5.0d0)+2.0d0
           vl(3) = ul*((4.0d0-3.0d0*ul)*ul+1.0d0)
           vl(4) = ul2*(ul-1.0d0)
           vl = 0.5d0*vl
           do ic=1,ncopies
              esxyz(ic) =  dot_product(vl,esyz(-1:2,ic))
              FRS(h,k,l,ic) = esxyz(ic)
           end do
        end do
     end do
  end do
  return
end subroutine tricubic

subroutine mtz2_3d(h,k,l,f,nobs,nx,ny,nz,f3d)
  implicit none
  integer,                        intent(in)  :: nx,ny,nz,nobs
  real,      dimension(nobs),     intent(in)  :: h,k,l
  complex*8, dimension(nobs),     intent(in)  :: f

  complex*8, dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2), intent(out)  :: f3d
  !complex*8, dimension(nx, ny, nz), intent(out)  :: f3d2
  !
  integer :: i, i1, i2, i3
  !
  f3d = 0.0

  do i = 1, nobs
     i1 = int(h(i))
     i2 = int(k(i))
     i3 = int(l(i))
     f3d(-i3,-i2,-i1)    = f(i) !Changed the axis order to comply with .mrc
     f3d(i3,i2,i1) = conjg(f(i))
     !if(i < 200) print*, i1,i2,i3,f3d(i1,i2,i3)     
  end do
  return
end subroutine mtz2_3d

subroutine prepare_hkl(hf1,nx,ny,nz,mode,h,k,l,ampli,phase)
  implicit none
  real*8,    parameter :: PI = 3.14159

  integer,                intent(in) :: mode,nx,ny,nz
  complex*8, dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(in)  :: hf1
  integer, dimension(nx*ny*ny/2),intent(out) :: h,k,l
  real, dimension(nx*ny*ny/2),intent(out) :: ampli, phase


  ! locals
  integer,   dimension(3)          :: nxyz
  integer    :: xyzmin(3),xyzmax(3)
  integer    :: i1,i2,i3,j
  real       :: hf1_real, hf1_imag
  logical    :: debug
  !
  debug         = .FALSE.
  if(mode == 1) debug = .TRUE.
  !call cpu_time(start)

  xyzmin = 0; xyzmax = 0

  nxyz = (/ nx, ny, nz /)

  h = 0; k = 0; l = 0


  xyzmin(1) = int(-nxyz(1)/2)
  xyzmin(2) = int(-nxyz(2)/2)
  xyzmin(3) = int(-nxyz(3)/2)
  xyzmax    = -(xyzmin+1)
  if(debug) print*, 'xyzmin = ', xyzmin
  if(debug) print*, 'xyzmax = ', xyzmax

  j = 0
  ! Not using Friedel's Law
  do i1=xyzmin(1), xyzmax(1)
     do i2=xyzmin(2), xyzmax(2)
        do i3=xyzmin(3), xyzmax(3)
           if(i3 < 0) cycle
           j = j + 1
           h(j) = i1
           k(j) = i2
           l(j) = i3
           hf1_real = real(hf1(i1,i2,i3))
           hf1_imag = aimag(hf1(i1,i2,i3))
           ampli(j) = sqrt(hf1_real**2 + hf1_imag**2)
           phase(j) = atan2(hf1_imag,hf1_real)*180.0/PI
        end do
     end do
  end do
  return
end subroutine prepare_hkl

subroutine add_random_phase_beyond(F_ori,F_all_random,uc,resol_randomize,nx,ny,nz,F_beyond_random)
  implicit none
  real*8,    parameter :: PI = 3.14159265359

  integer,   intent(in) :: nx,ny,nz
  real,      intent(in) :: resol_randomize
  real,      dimension(6),                                             intent(in)  :: uc
  complex*8, dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(in)  :: F_ori,F_all_random
  complex*8, dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(out) :: F_beyond_random
  ! locals
  integer,   dimension(3)          :: nxyz
  real       :: resol
  integer    :: xyzmin(3),xyzmax(3)
  integer    :: i1,i2,i3

  xyzmin = 0; xyzmax = 0

  nxyz = (/ nx, ny, nz /)

  F_beyond_random = 0.0
  

  xyzmin(1) = int(-nxyz(1)/2)
  xyzmin(2) = int(-nxyz(2)/2)
  xyzmin(3) = int(-nxyz(3)/2)
  xyzmax    = -(xyzmin+1)
  print*, 'xyzmin = ', xyzmin
  print*, 'xyzmax = ', xyzmax

  do i1=xyzmin(1), xyzmax(1)
     do i2=xyzmin(2), xyzmax(2)
        do i3=xyzmin(3), xyzmax(3)
           call get_resol(uc,real(i1),real(i2),real(i3),resol)
           if(resol >= resol_randomize)then
              F_beyond_random(i1,i2,i3) = F_ori(i1,i2,i3)
           else
              F_beyond_random(i1,i2,i3) = F_all_random(i1,i2,i3)
           end if
        end do
     end do
  end do
  return
end subroutine add_random_phase_beyond

subroutine cutmap(fin,bin_idx,smax,mode,nbin,nx,ny,nz,fout)
  implicit none
  integer,   intent(in) :: smax,mode,nbin,nx,ny,nz
  complex*8, dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(in)  :: fin
  integer,   dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(in)  :: bin_idx
  !
  complex*8, dimension(-nx/2:(nx-2)/2, -ny/2:(ny-2)/2, -nz/2:(nz-2)/2),intent(out) :: fout
  ! locals
  integer,   dimension(3) :: nxyz
  integer,   dimension(0:nbin-1) :: bin_arr_count
  !
  real       :: start,finish
  integer    :: i,j,k,n,xyzmin(3),xyzmax(3)
  logical    :: debug
  !
  debug         = .FALSE.
  if(mode == 1) debug = .TRUE.
  call cpu_time(start)

  fout = 0.0

  xyzmin = 0; xyzmax = 0

  nxyz = (/ nx, ny, nz /)

  xyzmin(1) = int(-nxyz(1)/2)
  xyzmin(2) = int(-nxyz(2)/2)
  xyzmin(3) = int(-nxyz(3)/2)
  xyzmax    = -(xyzmin+1)
  if(debug) print*, 'xyzmin = ', xyzmin
  if(debug) print*, 'xyzmax = ', xyzmax
  if(debug) print*, 'nbin=', nbin

  ! Using Friedel's Law
  do i=xyzmin(1), xyzmax(1)
     do j=xyzmin(2), xyzmax(2)
        do k=xyzmin(3), xyzmax(3)
           n = n + 1
           if(bin_idx(i,j,k) < 0 .or. bin_idx(i,j,k) > nbin-1) cycle
           if(bin_idx(i,j,k) > smax) cycle
           bin_arr_count(bin_idx(i,j,k)) = bin_arr_count(bin_idx(i,j,k)) + 1
           fout(i,j,k) = fin(i,j,k)
        end do
     end do
  end do

  call cpu_time(finish)
  if(debug) print*, 'time for Eo calculation(s) = ', finish-start
end subroutine cutmap

