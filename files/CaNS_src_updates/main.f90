! -
!
! SPDX-FileCopyrightText: Copyright (c) 2017-2022 Pedro Costa and the CaNS contributors. All rights reserved.
! SPDX-License-Identifier: MIT
!
! -
!
!        CCCCCCCCCCCCC                    NNNNNNNN        NNNNNNNN    SSSSSSSSSSSSSSS
!     CCC::::::::::::C                    N:::::::N       N::::::N  SS:::::::::::::::S
!   CC:::::::::::::::C                    N::::::::N      N::::::N S:::::SSSSSS::::::S
!  C:::::CCCCCCCC::::C                    N:::::::::N     N::::::N S:::::S     SSSSSSS
! C:::::C       CCCCCC   aaaaaaaaaaaaa    N::::::::::N    N::::::N S:::::S
!C:::::C                 a::::::::::::a   N:::::::::::N   N::::::N S:::::S
!C:::::C                 aaaaaaaaa:::::a  N:::::::N::::N  N::::::N  S::::SSSS
!C:::::C                          a::::a  N::::::N N::::N N::::::N   SS::::::SSSSS
!C:::::C                   aaaaaaa:::::a  N::::::N  N::::N:::::::N     SSS::::::::SS
!C:::::C                 aa::::::::::::a  N::::::N   N:::::::::::N        SSSSSS::::S
!C:::::C                a::::aaaa::::::a  N::::::N    N::::::::::N             S:::::S
! C:::::C       CCCCCC a::::a    a:::::a  N::::::N     N:::::::::N             S:::::S
!  C:::::CCCCCCCC::::C a::::a    a:::::a  N::::::N      N::::::::N SSSSSSS     S:::::S
!   CC:::::::::::::::C a:::::aaaa::::::a  N::::::N       N:::::::N S::::::SSSSSS:::::S
!     CCC::::::::::::C  a::::::::::aa:::a N::::::N        N::::::N S:::::::::::::::SS
!        CCCCCCCCCCCCC   aaaaaaaaaa  aaaa NNNNNNNN         NNNNNNN  SSSSSSSSSSSSSSS
!-------------------------------------------------------------------------------------
! CaNS -- Canonical Navier-Stokes Solver
!-------------------------------------------------------------------------------------

! SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: BSD-3-Clause
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! 3. Neither the name of the copyright holder nor the names of its
!    contributors may be used to endorse or promote products derived from
!    this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

program cans
#if defined(_DEBUG)
  use, intrinsic :: iso_fortran_env, only: compiler_version,compiler_options
#endif
  use, intrinsic :: iso_c_binding  , only: C_PTR
  use, intrinsic :: ieee_arithmetic, only: is_nan => ieee_is_nan
  use mpi
  use decomp_2d
  use torchfort
#if defined(_OPENACC)
  use cudafor
#endif
  use mod_bound          , only: boundp,bounduvw,updt_rhs_b
  use mod_chkdiv         , only: chkdiv
  use mod_chkdt          , only: chkdt
  use mod_common_mpi     , only: myid,ierr
  use mod_correc         , only: correc
  use mod_fft            , only: fftini,fftend
  use mod_fillps         , only: fillps
  use mod_initflow       , only: initflow
  use mod_initgrid       , only: initgrid
  use mod_initmpi        , only: initmpi
  use mod_initsolver     , only: initsolver
  use mod_load           , only: load_all
  use mod_mom            , only: bulk_forcing
  use mod_rk             , only: rk
  use mod_output         , only: out0d,gen_alias,out1d,out1d_chan,out2d,out3d,write_log_output,write_visu_2d,write_visu_3d
  use mod_param          , only: ng,l,dl,dli, &
                                 gtype,gr, &
                                 cfl,dtmin, &
                                 visc, &
                                 inivel,is_wallturb, &
                                 nstep,time_max,tw_max,stop_type, &
                                 restart,is_overwrite_save,nsaves_max, &
                                 icheck,iout0d,iout1d,iout2d,iout3d,isave, &
                                 cbcvel,bcvel,cbcpre,bcpre, &
                                 is_forced,bforce,velf, &
                                 dims, &
                                 nb,is_bound, &
                                 rkcoeff,small, &
                                 datadir,   &
                                 read_input, &
                                 model_name, model_config_file, twarmup, tmean, dtsample, &
                                 trainbs, read_input, yplus, nsamples_train, nsamples_val, &
                                 torchfort_load_ckpt, torchfort_ckpt, test_inference, &
                                 nranks
  use mod_sanity         , only: test_sanity_input,test_sanity_solver
#if !defined(_OPENACC)
  use mod_solver         , only: solver
#if defined(_IMPDIFF_1D)
  use mod_solver         , only: solver_gaussel_z
#endif
#else
  use mod_solver_gpu     , only: solver => solver_gpu
#if defined(_IMPDIFF_1D)
  use mod_solver_gpu     , only: solver_gaussel_z => solver_gaussel_z_gpu
#endif
  use mod_workspaces     , only: init_wspace_arrays,set_cufft_wspace
  use mod_common_cudecomp, only: istream_acc_queue_1
#endif
  use mod_timer          , only: timer_tic,timer_toc,timer_print
  use mod_updatep        , only: updatep
  use mod_utils          , only: bulk_mean
  !@acc use mod_utils    , only: device_memory_footprint
  use mod_types
  use omp_lib
  use mod_dlutils        , only: cmpt_wallstresses_2d, write_sample, distribute_batches
  implicit none
  integer , dimension(3) :: lo,hi,n,n_x_fft,n_y_fft,lo_z,hi_z,n_z
  real(rp), allocatable, dimension(:,:,:) :: u,v,w,p,pp
  real(rp), allocatable, dimension(:,:,:,:) :: input_local, label_local
  real(rp), allocatable, dimension(:,:,:,:) :: input, output, label
  real(rp), allocatable, dimension(:,:,:) :: tauxz, tauyz, tauzz
  real(rp) :: loss_value
  real(8), allocatable, dimension(:) :: umean, vmean, wmean
  real(8), allocatable, dimension(:) :: ustd, vstd, wstd
  real(8):: umean_inf, vmean_inf, wmean_inf
  real(8):: ustd_inf, vstd_inf, wstd_inf
  real(8), allocatable, dimension(:) :: txzmean, tyzmean, tzzmean
  real(8), allocatable, dimension(:) :: txzstd, tyzstd, tzzstd
  real(rp), dimension(3) :: tauxo,tauyo,tauzo
  real(rp), dimension(3) :: f
#if !defined(_OPENACC)
  type(C_PTR), dimension(2,2) :: arrplanp
#else
  integer    , dimension(2,2) :: arrplanp
#endif
  real(rp), allocatable, dimension(:,:) :: lambdaxyp
  real(rp), allocatable, dimension(:) :: ap,bp,cp
  real(rp) :: normfftp
  type rhs_bound
    real(rp), allocatable, dimension(:,:,:) :: x
    real(rp), allocatable, dimension(:,:,:) :: y
    real(rp), allocatable, dimension(:,:,:) :: z
  end type rhs_bound
  type(rhs_bound) :: rhsbp
  real(rp) :: alpha
#if defined(_IMPDIFF)
#if !defined(_OPENACC)
  type(C_PTR), dimension(2,2) :: arrplanu,arrplanv,arrplanw
#else
  integer    , dimension(2,2) :: arrplanu,arrplanv,arrplanw
#endif
  real(rp), allocatable, dimension(:,:) :: lambdaxyu,lambdaxyv,lambdaxyw,lambdaxy
  real(rp), allocatable, dimension(:) :: au,av,aw,bu,bv,bw,cu,cv,cw,aa,bb,cc
  real(rp) :: normfftu,normfftv,normfftw
  type(rhs_bound) :: rhsbu,rhsbv,rhsbw
  real(rp), allocatable, dimension(:,:,:) :: rhsbx,rhsby,rhsbz
#endif
  real(rp) :: dt,dti,dtmax,time,dtrk,dtrki,divtot,divmax
  integer :: irk,istep
  real(rp), allocatable, dimension(:) :: dzc  ,dzf  ,zc  ,zf  ,dzci  ,dzfi, &
                                         dzc_g,dzf_g,zc_g,zf_g,dzci_g,dzfi_g, &
                                         grid_vol_ratio_c,grid_vol_ratio_f
  real(rp) :: meanvelu,meanvelv,meanvelw
  real(rp), dimension(3) :: dpdl
  !real(rp), allocatable, dimension(:) :: var
  real(rp), dimension(42) :: var
#if defined(_TIMING)
  real(rp) :: dt12,dt12av,dt12min,dt12max
#endif
  real(rp) :: twi,tw
  integer  :: savecounter
  character(len=7  ) :: fldnum
  character(len=7  ) :: trainckptnum
  character(len=4  ) :: chkptnum
  character(len=100) :: filename
  integer :: k,kk
  logical :: is_done,kill
  integer :: ktop, kbot
  real(rp) :: rey, retau, timesample, dtsample_scaled, timevis, dtvis_scaled
  integer :: isteptrain, istepval, ii, istat
  integer :: statcount, sampleidx
  real(rp):: stattime
  real(rp):: tmp
  logical :: is_training, is_validating
  real(rp) :: uerr, verr, werr
  integer :: dev
  !
  call MPI_INIT(ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,nranks,ierr)
  !
  ! read parameter file
  !
  call read_input(myid)
  !
  ! initialize MPI/OpenMP
  !
  !$ call omp_set_num_threads(omp_get_max_threads())
  call initmpi(ng,dims,cbcpre,lo,hi,n,n_x_fft,n_y_fft,lo_z,hi_z,n_z,nb,is_bound)
  !$acc enter data copyin(ng)
  twi = MPI_WTIME()
  savecounter = 0
  !
  ! allocate variables
  !
  allocate(u( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
           v( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
           w( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
           p( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
           pp(0:n(1)+1,0:n(2)+1,0:n(3)+1))
  ! CHECKME: NN related stuff
  ! do not transpose the spatial dims, not necessary. fix later
  allocate( input_local(1:n(1), 1:n(2), 1:3, trainbs * nranks), &
            label_local(1:n(1), 1:n(2), 1:3, trainbs * nranks))
  !$acc enter data create(input_local, label_local)
  allocate( input(1:ng(1), 1:ng(2), 1:3, trainbs), &
           output(1:ng(1), 1:ng(2), 1:3, trainbs), &
            label(1:ng(1), 1:ng(2), 1:3, trainbs))
  !$acc enter data create(input, output, label)
  allocate(tauxz(1:n(1), 1:n(2), 0:1), &
           tauyz(1:n(1), 1:n(2), 0:1), &
           tauzz(1:n(1), 1:n(2), 0:1))
  !$acc enter data create(tauxz, tauyz, tauzz)
  !
  allocate(lambdaxyp(n_z(1),n_z(2)))
  allocate(ap(n_z(3)),bp(n_z(3)),cp(n_z(3)))
  allocate(dzc( 0:n(3)+1), &
           dzf( 0:n(3)+1), &
           zc(  0:n(3)+1), &
           zf(  0:n(3)+1), &
           dzci(0:n(3)+1), &
           dzfi(0:n(3)+1))
  allocate(dzc_g( 0:ng(3)+1), &
           dzf_g( 0:ng(3)+1), &
           zc_g(  0:ng(3)+1), &
           zf_g(  0:ng(3)+1), &
           dzci_g(0:ng(3)+1), &
           dzfi_g(0:ng(3)+1))
  allocate(grid_vol_ratio_c,mold=dzc)
  allocate(grid_vol_ratio_f,mold=dzf)
  allocate(rhsbp%x(n(2),n(3),0:1), &
           rhsbp%y(n(1),n(3),0:1), &
           rhsbp%z(n(1),n(2),0:1))
#if defined(_IMPDIFF)
  allocate(lambdaxyu(n_z(1),n_z(2)), &
           lambdaxyv(n_z(1),n_z(2)), &
           lambdaxyw(n_z(1),n_z(2)), &
           lambdaxy( n_z(1),n_z(2)))
  allocate(au(n_z(3)),bu(n_z(3)),cu(n_z(3)), &
           av(n_z(3)),bv(n_z(3)),cv(n_z(3)), &
           aw(n_z(3)),bw(n_z(3)),cw(n_z(3)), &
           aa(n_z(3)),bb(n_z(3)),cc(n_z(3)))
  allocate(rhsbu%x(n(2),n(3),0:1), &
           rhsbu%y(n(1),n(3),0:1), &
           rhsbu%z(n(1),n(2),0:1), &
           rhsbv%x(n(2),n(3),0:1), &
           rhsbv%y(n(1),n(3),0:1), &
           rhsbv%z(n(1),n(2),0:1), &
           rhsbw%x(n(2),n(3),0:1), &
           rhsbw%y(n(1),n(3),0:1), &
           rhsbw%z(n(1),n(2),0:1), &
           rhsbx(  n(2),n(3),0:1), &
           rhsby(  n(1),n(3),0:1), &
           rhsbz(  n(1),n(2),0:1))
#endif

  ! torchfort: create model
#if defined(_OPENACC)
  istat = cudaGetDevice(dev)
  if (nranks == 1) then
    istat = torchfort_create_model(model_name, model_config_file, dev)
  else
    istat = torchfort_create_distributed_model(model_name, model_config_file, MPI_COMM_WORLD, dev)
  endif
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop
#else
  if (nranks == 1) then
    istat = torchfort_create_model(model_name, model_config_file, TORCHFORT_DEVICE_CPU)
  else
    istat = torchfort_create_distributed_model(model_name, model_config_file, MPI_COMM_WORLD, TORCHFORT_DEVICE_CPU)
  endif
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop
#endif

  if (torchfort_load_ckpt) then
    if (myid == 0) print*, "Loading torchfort checkpoint", torchfort_ckpt
    istat = torchfort_load_checkpoint(model_name, torchfort_ckpt, isteptrain, istepval)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    if (myid == 0) print*, "isteptrain", isteptrain, "istepval", istepval
  else
    isteptrain = 0
    istepval = 0
  endif
  
#if defined(_DEBUG)
  if(myid == 0) print*, 'This executable of CaNS was built with compiler: ', compiler_version()
  if(myid == 0) print*, 'Using the options: ', compiler_options()
  block
    character(len=MPI_MAX_LIBRARY_VERSION_STRING) :: mpi_version
    integer :: ilen
    call MPI_GET_LIBRARY_VERSION(mpi_version,ilen,ierr)
    if(myid == 0) print*, 'MPI Version: ', trim(mpi_version)
  end block
  if(myid == 0) print*, ''
#endif
  if(myid == 0) print*, '*******************************'
  if(myid == 0) print*, '*** Beginning of simulation ***'
  if(myid == 0) print*, '*******************************'
  if(myid == 0) print*, ''
  call initgrid(gtype,ng(3),gr,l(3),dzc_g,dzf_g,zc_g,zf_g)
  if(myid == 0) then
    open(99,file=trim(datadir)//'grid.bin',action='write',form='unformatted',access='stream',status='replace')
    write(99) dzc_g(1:ng(3)),dzf_g(1:ng(3)),zc_g(1:ng(3)),zf_g(1:ng(3))
    close(99)
    open(99,file=trim(datadir)//'grid.out')
    do kk=0,ng(3)+1
      write(99,*) 0.,zf_g(kk),zc_g(kk),dzf_g(kk),dzc_g(kk)
    end do
    close(99)
    open(99,file=trim(datadir)//'geometry.out')
      write(99,*) ng(1),ng(2),ng(3)
      write(99,*) l(1),l(2),l(3)
    close(99)
  end if
  !$acc enter data copyin(lo,hi,n) async
  !$acc enter data copyin(bforce,dl,dli,l) async
  !$acc enter data copyin(zc_g,zf_g,dzc_g,dzf_g) async
  !$acc enter data create(zc,zf,dzc,dzf,dzci,dzfi,dzci_g,dzfi_g) async
  !
  !$acc parallel loop default(present) private(k) async
  do kk=lo(3)-1,hi(3)+1
    k = kk-(lo(3)-1)
    zc( k) = zc_g(kk)
    zf( k) = zf_g(kk)
    dzc(k) = dzc_g(kk)
    dzf(k) = dzf_g(kk)
    dzci(k) = dzc(k)**(-1)
    dzfi(k) = dzf(k)**(-1)
  end do
  !$acc kernels default(present) async
  dzci_g(:) = dzc_g(:)**(-1)
  dzfi_g(:) = dzf_g(:)**(-1)
  !$acc end kernels
  !$acc enter data create(grid_vol_ratio_c,grid_vol_ratio_f) async
  !$acc kernels default(present) async
  grid_vol_ratio_c(:) = dl(1)*dl(2)*dzc(:)/(l(1)*l(2)*l(3))
  grid_vol_ratio_f(:) = dl(1)*dl(2)*dzf(:)/(l(1)*l(2)*l(3))
  !$acc end kernels
  !$acc update self(zc,zf,dzc,dzf,dzci,dzfi) async
  !$acc exit data copyout(zc_g,zf_g,dzc_g,dzf_g,dzci_g,dzfi_g) async ! not needed on the device
  !$acc wait
  !
  ! test input files before proceeding with the calculation
  !
  call test_sanity_input(ng,dims,stop_type,cbcvel,cbcpre,bcvel,bcpre,is_forced)
  !
  ! initialize Poisson solver
  !
  call initsolver(ng,n_x_fft,n_y_fft,lo_z,hi_z,dli,dzci_g,dzfi_g,cbcpre,bcpre(:,:), &
                  lambdaxyp,['c','c','c'],ap,bp,cp,arrplanp,normfftp,rhsbp%x,rhsbp%y,rhsbp%z)
  !$acc enter data copyin(lambdaxyp,ap,bp,cp) async
  !$acc enter data copyin(rhsbp,rhsbp%x,rhsbp%y,rhsbp%z) async
  !$acc wait
#if defined(_IMPDIFF)
  call initsolver(ng,n_x_fft,n_y_fft,lo_z,hi_z,dli,dzci_g,dzfi_g,cbcvel(:,:,1),bcvel(:,:,1), &
                  lambdaxyu,['f','c','c'],au,bu,cu,arrplanu,normfftu,rhsbu%x,rhsbu%y,rhsbu%z)
  call initsolver(ng,n_x_fft,n_y_fft,lo_z,hi_z,dli,dzci_g,dzfi_g,cbcvel(:,:,2),bcvel(:,:,2), &
                  lambdaxyv,['c','f','c'],av,bv,cv,arrplanv,normfftv,rhsbv%x,rhsbv%y,rhsbv%z)
  call initsolver(ng,n_x_fft,n_y_fft,lo_z,hi_z,dli,dzci_g,dzfi_g,cbcvel(:,:,3),bcvel(:,:,3), &
                  lambdaxyw,['c','c','f'],aw,bw,cw,arrplanw,normfftw,rhsbw%x,rhsbw%y,rhsbw%z)
#if defined(_IMPDIFF_1D)
  deallocate(lambdaxyu,lambdaxyv,lambdaxyw,lambdaxy)
  call fftend(arrplanu)
  call fftend(arrplanv)
  call fftend(arrplanw)
  deallocate(rhsbu%x,rhsbu%y,rhsbv%x,rhsbv%y,rhsbw%x,rhsbw%y,rhsbx,rhsby)
#endif
  !$acc enter data copyin(lambdaxyu,au,bu,cu,lambdaxyv,av,bv,cv,lambdaxyw,aw,bw,cw) async
  !$acc enter data copyin(rhsbu,rhsbu%x,rhsbu%y,rhsbu%z) async
  !$acc enter data copyin(rhsbv,rhsbv%x,rhsbv%y,rhsbv%z) async
  !$acc enter data copyin(rhsbw,rhsbw%x,rhsbw%y,rhsbw%z) async
  !$acc enter data create(lambdaxy,aa,bb,cc) async
  !$acc enter data create(rhsbx,rhsby,rhsbz) async
  !$acc wait
#endif
#if defined(_OPENACC)
  !
  ! determine workspace sizes and allocate the memory
  !
  call init_wspace_arrays()
  call set_cufft_wspace(pack(arrplanp,.true.),istream_acc_queue_1)
#if defined(_IMPDIFF) && !defined(_IMPDIFF_1D)
  call set_cufft_wspace(pack(arrplanu,.true.),istream_acc_queue_1)
  call set_cufft_wspace(pack(arrplanv,.true.),istream_acc_queue_1)
  call set_cufft_wspace(pack(arrplanw,.true.),istream_acc_queue_1)
#endif
  if(myid == 0) print*,'*** Device memory footprint (Gb): ', &
                  device_memory_footprint(n,n_z)/(1._sp*1024**3), ' ***'
#endif
#if defined(_DEBUG_SOLVER)
  call test_sanity_solver(ng,lo,hi,n,n_x_fft,n_y_fft,lo_z,hi_z,n_z,dli,dzc,dzf,dzci,dzfi,dzci_g,dzfi_g, &
                          nb,is_bound,cbcvel,cbcpre,bcvel,bcpre)
#endif
  !
  if(.not.restart) then
    istep = 0
    time = 0.
    call initflow(inivel,bcvel,ng,lo,l,dl,zc,zf,dzc,dzf,visc, &
                  is_forced,velf,bforce,is_wallturb,u,v,w,p)
    if(myid == 0) print*, '*** Initial condition succesfully set ***'
  else
    call load_all('r',trim(datadir)//'fld.bin',MPI_COMM_WORLD,ng,[1,1,1],lo,hi,u,v,w,p,time,istep)
    if(myid == 0) print*, '*** Checkpoint loaded at time = ', time, 'time step = ', istep, '. ***'
  end if
  !$acc enter data copyin(u,v,w,p) create(pp)
  call bounduvw(cbcvel,n,bcvel,nb,is_bound,.false.,dl,dzc,dzf,u,v,w)
  call boundp(cbcpre,n,bcpre,nb,is_bound,dl,dzc,p)
  !
  ! post-process and write initial condition
  !
  write(fldnum,'(i7.7)') istep
  !$acc wait ! not needed but to prevent possible future issues
  !$acc update self(u,v,w,p)
  include 'out1d.h90'
  include 'out2d.h90'
  include 'out3d.h90'
  !
  call chkdt(n,dl,dzci,dzfi,visc,u,v,w,dtmax)
  dt = min(cfl*dtmax,dtmin)
  if(myid == 0) print*, 'dtmax = ', dtmax, 'dt = ', dt
  dti = 1./dt
  kill = .false.

  ! CHECKME: compute some NN related parameters
  rey = 5640.
  retau = 0.09*rey**0.88
  !do ktop = 0, n(3)+1
  !  if (zc(ktop) .ge. 15/retau) exit
  !enddo
  !do kbot = 0, n(3)+1
  !  if (lz-zc(kbot) .le. 15/retau) exit
  !enddo
  !kbot = kbot-1
  !$acc update self(zc)
  print*, "yplus", yplus 
  kbot = minloc(abs(retau*(   zc(1:n(3)))-yplus),1)
  ktop = minloc(abs(retau*(l(3)-zc(1:n(3)))-yplus),1)
  print*, 'ktop = ', ktop, 'kbot = ', kbot
  dtsample_scaled = dtsample / (retau**2 / (rey / 2))
  print*, 'dtsample (scaled) = ', dtsample_scaled
  !dtvis_scaled = dtvis / (retau**2 / (rey / 2))
  !print*, 'dtvis (scaled) = ', dtvis_scaled
  !
  ! main loop
  !
  if(myid == 0) print*, '*** Calculation loop starts now ***'
  is_done = .false.
  is_training = .false.
  is_validating = .false.
  timesample = 0.
  timevis = 0.
  statcount = 0
  stattime = 0.
  sampleidx = 0

  ! arrays for statistics
  allocate(umean(n(3)), vmean(n(3)), wmean(n(3)))
  allocate(ustd(n(3)), vstd(n(3)), wstd(n(3)))
  allocate(txzmean(2), tyzmean(2), tzzmean(2))
  allocate(txzstd(2), tyzstd(2), tzzstd(2))
  umean = 0
  vmean = 0
  wmean = 0
  ustd = 0
  vstd = 0
  wstd = 0
  txzmean = 0
  tyzmean = 0
  tzzmean = 0
  txzstd = 0
  tyzstd = 0
  tzzstd = 0
  umean_inf = 0
  vmean_inf = 0
  wmean_inf = 0
  ustd_inf = 0
  vstd_inf = 0
  wstd_inf = 0

  if (restart .or. test_inference) then
    print*, "Reading normalization stats..."
    ! read normalization stats from file
    open(42, status='old',action='read', file="data/velstats.txt")
    do ii = 1, n(3)
      read(42, *), tmp, umean(ii), vmean(ii), wmean(ii), ustd(ii), vstd(ii), wstd(ii)
    enddo
    close(42)
    open(42, status='old',action='read', file="data/taustats.txt")
    do ii = 1, 2
      read(42, *),  txzmean(ii), tyzmean(ii), tzzmean(ii), txzstd(ii), tyzstd(ii), tzzstd(ii)
    enddo
    close(42)

    if (.not. test_inference) then
      is_training = .true.
    endif
  endif

  !$acc enter data copyin(umean, vmean, wmean)
  !$acc enter data copyin(ustd, vstd, wstd)
  !$acc enter data copyin(txzmean, tyzmean, tzzmean)
  !$acc enter data copyin(txzstd, tyzstd, tzzstd)

  do while(.not.is_done)
#if defined(_TIMING)
    !$acc wait(1)
    dt12 = MPI_WTIME()
#endif
    istep = istep + 1
    time = time + dt
    timesample = timesample + dt
    timevis = timevis + dt
    if(myid == 0 .and. mod(istep, 100) == 0) print*, 'Time step #', istep, 'Time = ', time

    !! check if we need to do training
    !if ((istep >= nwarmup) .and. (istep < ntraining + nwarmup)) is_training = .true.
    
    tauxo(:) = 0.
    tauyo(:) = 0.
    tauzo(:) = 0.
    dpdl(:)  = 0.
    do irk=1,3
      dtrk = sum(rkcoeff(:,irk))*dt
      dtrki = dtrk**(-1)
      call rk(rkcoeff(:,irk),n,dli,dzci,dzfi,grid_vol_ratio_c,grid_vol_ratio_f,visc,dt,p, &
              is_forced,velf,bforce,u,v,w,f)
      call bulk_forcing(n,is_forced,f,u,v,w)
#if defined(_IMPDIFF)
      alpha = -.5*visc*dtrk
      !$OMP PARALLEL WORKSHARE
      !$acc kernels present(rhsbx,rhsby,rhsbz,rhsbu) async(1)
#if !defined(_IMPDIFF_1D)
      rhsbx(:,:,0:1) = rhsbu%x(:,:,0:1)*alpha
      rhsby(:,:,0:1) = rhsbu%y(:,:,0:1)*alpha
#endif
      rhsbz(:,:,0:1) = rhsbu%z(:,:,0:1)*alpha
      !$acc end kernels
      !$OMP END PARALLEL WORKSHARE
      call updt_rhs_b(['f','c','c'],cbcvel(:,:,1),n,is_bound,rhsbx,rhsby,rhsbz,u)
      !$acc kernels default(present) async(1)
      !$OMP PARALLEL WORKSHARE
      aa(:) = au(:)*alpha
      bb(:) = bu(:)*alpha + 1.
      cc(:) = cu(:)*alpha
#if !defined(_IMPDIFF_1D)
      lambdaxy(:,:) = lambdaxyu(:,:)*alpha
#endif
      !$OMP END PARALLEL WORKSHARE
      !$acc end kernels
#if !defined(_IMPDIFF_1D)
      call solver(n,ng,arrplanu,normfftu,lambdaxy,aa,bb,cc,cbcvel(:,:,1),['f','c','c'],u)
#else
      call solver_gaussel_z(n                    ,aa,bb,cc,cbcvel(:,3,1),['f','c','c'],u)
#endif
      !$OMP PARALLEL WORKSHARE
      !$acc kernels present(rhsbx,rhsby,rhsbz,rhsbv) async(1)
#if !defined(_IMPDIFF_1D)
      rhsbx(:,:,0:1) = rhsbv%x(:,:,0:1)*alpha
      rhsby(:,:,0:1) = rhsbv%y(:,:,0:1)*alpha
#endif
      rhsbz(:,:,0:1) = rhsbv%z(:,:,0:1)*alpha
      !$acc end kernels
      !$OMP END PARALLEL WORKSHARE
      call updt_rhs_b(['c','f','c'],cbcvel(:,:,2),n,is_bound,rhsbx,rhsby,rhsbz,v)
      !$acc kernels default(present) async(1)
      !$OMP PARALLEL WORKSHARE
      aa(:) = av(:)*alpha
      bb(:) = bv(:)*alpha + 1.
      cc(:) = cv(:)*alpha
#if !defined(_IMPDIFF_1D)
      lambdaxy(:,:) = lambdaxyv(:,:)*alpha
#endif
      !$OMP END PARALLEL WORKSHARE
      !$acc end kernels
#if !defined(_IMPDIFF_1D)
      call solver(n,ng,arrplanv,normfftv,lambdaxy,aa,bb,cc,cbcvel(:,:,2),['c','f','c'],v)
#else
      call solver_gaussel_z(n                    ,aa,bb,cc,cbcvel(:,3,2),['c','f','c'],v)
#endif
      !$OMP PARALLEL WORKSHARE
      !$acc kernels present(rhsbx,rhsby,rhsbz,rhsbw) async(1)
#if !defined(_IMPDIFF_1D)
      rhsbx(:,:,0:1) = rhsbw%x(:,:,0:1)*alpha
      rhsby(:,:,0:1) = rhsbw%y(:,:,0:1)*alpha
#endif
      rhsbz(:,:,0:1) = rhsbw%z(:,:,0:1)*alpha
      !$acc end kernels
      !$OMP END PARALLEL WORKSHARE
      call updt_rhs_b(['c','c','f'],cbcvel(:,:,3),n,is_bound,rhsbx,rhsby,rhsbz,w)
      !$acc kernels default(present) async(1)
      !$OMP PARALLEL WORKSHARE
      aa(:) = aw(:)*alpha
      bb(:) = bw(:)*alpha + 1.
      cc(:) = cw(:)*alpha
#if !defined(_IMPDIFF_1D)
      lambdaxy(:,:) = lambdaxyw(:,:)*alpha
#endif
      !$OMP END PARALLEL WORKSHARE
      !$acc end kernels
#if !defined(_IMPDIFF_1D)
      call solver(n,ng,arrplanw,normfftw,lambdaxy,aa,bb,cc,cbcvel(:,:,3),['c','c','f'],w)
#else
      call solver_gaussel_z(n                    ,aa,bb,cc,cbcvel(:,3,3),['c','c','f'],w)
#endif
#endif
      dpdl(:) = dpdl(:) + f(:)
      call bounduvw(cbcvel,n,bcvel,nb,is_bound,.false.,dl,dzc,dzf,u,v,w)
      call fillps(n,dli,dzfi,dtrki,u,v,w,pp)
      call updt_rhs_b(['c','c','c'],cbcpre,n,is_bound,rhsbp%x,rhsbp%y,rhsbp%z,pp)
      call solver(n,ng,arrplanp,normfftp,lambdaxyp,ap,bp,cp,cbcpre,['c','c','c'],pp)
      call boundp(cbcpre,n,bcpre,nb,is_bound,dl,dzc,pp)
      call correc(n,dli,dzci,dtrk,pp,u,v,w)
      call bounduvw(cbcvel,n,bcvel,nb,is_bound,.true.,dl,dzc,dzf,u,v,w)
      call updatep(n,dli,dzci,dzfi,alpha,pp,p)
      call boundp(cbcpre,n,bcpre,nb,is_bound,dl,dzc,p)
    end do
    dpdl(:) = -dpdl(:)*dti

    ! CHECKME: start gathering statistics
    if (.not. restart .or. test_inference) then
      stattime = stattime + dt
      if (time > twarmup .and. time < twarmup + tmean .and. stattime >= dtsample_scaled) then
        if (myid == 0 .and. statcount == 0) print*, "warmup complete. begin computing means..."

        call cmpt_wallstresses_2d(n, is_bound, dzci, visc, u, v, p, &
                                  tauxz, tauyz, tauzz)

        if (.not. test_inference) then
          !$acc kernels default(present)
          umean = umean + sum(sum(u(1:n(1), 1:n(2), 1:n(3)), 1),1) / (ng(1) * ng(2))
          vmean = vmean + sum(sum(v(1:n(1), 1:n(2), 1:n(3)), 1),1) / (ng(1) * ng(2))
          wmean = wmean + sum(sum(w(1:n(1), 1:n(2), 1:n(3)), 1),1) / (ng(1) * ng(2))

          ustd = ustd + sum(sum(u(1:n(1), 1:n(2), 1:n(3))**2, 1),1) / (ng(1) * ng(2))
          vstd = vstd + sum(sum(v(1:n(1), 1:n(2), 1:n(3))**2, 1),1) / (ng(1) * ng(2))
          wstd = wstd + sum(sum(w(1:n(1), 1:n(2), 1:n(3))**2, 1),1) / (ng(1) * ng(2))

          txzmean = txzmean + sum(sum(tauxz(:,:,0:1), 1),1) / (ng(1) * ng(2))
          tyzmean = tyzmean + sum(sum(tauyz(:,:,0:1), 1),1) / (ng(1) * ng(2))
          tzzmean = tzzmean + sum(sum(tauzz(:,:,0:1), 1),1) / (ng(1) * ng(2))

          txzstd = txzstd + sum(sum(tauxz(:,:,0:1)**2, 1),1) / (ng(1) * ng(2))
          tyzstd = tyzstd + sum(sum(tauyz(:,:,0:1)**2, 1),1) / (ng(1) * ng(2))
          tzzstd = tzzstd + sum(sum(tauzz(:,:,0:1)**2, 1),1) / (ng(1) * ng(2))
          !$acc end kernels
          statcount  = statcount + 1
        endif

        sampleidx = sampleidx + 1
        stattime = 0
        if (test_inference) then
          !$acc kernels default(present)
          input_local(1:n(1), 1:n(2), 1, sampleidx) = (tauxz(1:n(1),1:n(2),0) - txzmean(1)) / txzstd(1)
          input_local(1:n(1), 1:n(2), 2, sampleidx) = (tauyz(1:n(1),1:n(2),0) - tyzmean(1)) / tyzstd(1)
          input_local(1:n(1), 1:n(2), 3, sampleidx) = (tauzz(1:n(1),1:n(2),0) - tzzmean(1)) / tzzstd(1)
          !$acc end kernels

          if (mod(sampleidx, trainbs * nranks) == 0) then
             sampleidx = 0
             call distribute_batches(input_local, input, n, ng)

             !$acc host_data use_device(input, output)
             istat = torchfort_inference(model_name, input, output)
             !$acc end host_data
             if (istat /= TORCHFORT_RESULT_SUCCESS) stop

             !$acc kernels default(present)
             ! Unapply normalization to get raw velocities
             output(1:ng(1), 1:ng(2), 1, :) = output(1:ng(1), 1:ng(2), 1, :) * ustd(kbot) + umean(kbot)
             output(1:ng(1), 1:ng(2), 2, :) = output(1:ng(1), 1:ng(2), 2, :) * vstd(kbot) + vmean(kbot)
             output(1:ng(1), 1:ng(2), 3, :) = output(1:ng(1), 1:ng(2), 3, :) * wstd(kbot) + wmean(kbot)

             ! Process batch stats
             umean_inf = umean_inf + sum(output(1:ng(1), 1:ng(2), 1, :)) / (ng(1) * ng(2) * trainbs * nranks)
             vmean_inf = vmean_inf + sum(output(1:ng(1), 1:ng(2), 2, :)) / (ng(1) * ng(2) * trainbs * nranks)
             wmean_inf = wmean_inf + sum(output(1:ng(1), 1:ng(2), 3, :)) / (ng(1) * ng(2) * trainbs * nranks)

             ustd_inf = ustd_inf + sum(output(1:ng(1), 1:ng(2), 1, :)**2) / (ng(1) * ng(2) * trainbs * nranks)
             vstd_inf = vstd_inf + sum(output(1:ng(1), 1:ng(2), 2, :)**2) / (ng(1) * ng(2) * trainbs * nranks)
             wstd_inf = wstd_inf + sum(output(1:ng(1), 1:ng(2), 3, :)**2) / (ng(1) * ng(2) * trainbs * nranks)
             !$acc end kernels

             statcount  = statcount + 1
          endif
        endif

      else if ((time >= twarmup + tmean) .and. (is_training .eqv. .false.)) then
        if (.not. test_inference) then
          !$acc update host(umean, vmean, wmean)
          !$acc update host(ustd, vstd, wstd)
          !$acc update host(txzmean, tyzmean, tzzmean)
          !$acc update host(txzstd, tyzstd, tzzstd)
          call MPI_ALLREDUCE(MPI_IN_PLACE,umean,ng(3),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,vmean,ng(3),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,wmean,ng(3),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,ustd,ng(3),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,vstd,ng(3),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,wstd,ng(3),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,txzmean,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,tyzmean,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,tzzmean,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,txzstd,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,tyzstd,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,tzzstd,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          !$acc update device(umean, vmean, wmean)
          !$acc update device(ustd, vstd, wstd)
          !$acc update device(txzmean, tyzmean, tzzmean)
          !$acc update device(txzstd, tyzstd, tzzstd)

          !$acc kernels default(present)
          umean = umean / statcount
          vmean = vmean / statcount
          wmean = wmean / statcount

          ustd = sqrt(ustd / statcount - umean**2)
          vstd = sqrt(vstd / statcount - vmean**2)
          wstd = sqrt(wstd / statcount - wmean**2)

          txzmean = txzmean / statcount
          tyzmean = tyzmean / statcount
          tzzmean = tzzmean / statcount

          txzstd = sqrt(txzstd / statcount - txzmean**2)
          tyzstd = sqrt(tyzstd / statcount - tyzmean**2)
          tzzstd = sqrt(tzzstd / statcount - tzzmean**2)
          !$acc end kernels

          if (myid == 0) then
            print*, "mean computation complete"
            print*, "statcount", statcount
            !$acc update host(umean, vmean, wmean)
            !$acc update host(txzmean, tyzmean, tzzmean)
            print*, "umean_top", umean(ktop)
            print*, "vmean_top", vmean(ktop)
            print*, "wmean_top", wmean(ktop)
            print*, "umean_bot", umean(kbot)
            print*, "vmean_bot", vmean(kbot)
            print*, "wmean_bot", wmean(kbot)
            print*, "txzmean_top", txzmean(2)
            print*, "tyzmean_top", tyzmean(2)
            print*, "tzzmean_top", tzzmean(2)
            print*, "txzmean_bot", txzmean(1)
            print*, "tyzmean_bot", tyzmean(1)
            print*, "tzzmean_bot", tzzmean(1)


            print*, "statistics computation complete"
            !$acc update host(ustd, vstd, wstd)
            !$acc update host(txzstd, tyzstd, tzzstd)
            print*, "ustd_top", ustd(ktop)
            print*, "vstd_top", vstd(ktop)
            print*, "wstd_top", wstd(ktop)
            print*, "ustd_bot", ustd(kbot)
            print*, "vstd_bot", vstd(kbot)
            print*, "wstd_bot", wstd(kbot)
            print*, "txzstd_top", txzstd(2)
            print*, "tyzstd_top", tyzstd(2)
            print*, "tzzstd_top", tzzstd(2)
            print*, "txzstd_bot", txzstd(1)
            print*, "tyzstd_bot", tyzstd(1)
            print*, "tzzstd_bot", tzzstd(1)

            open(42, file="data/velstats.txt")
            do ii = 1, n(3)
              write(42, '(*(E16.7e3))'), zc(ii), umean(ii), vmean(ii), wmean(ii), ustd(ii), vstd(ii), wstd(ii)
            enddo
            close(42)
            open(42, file="data/taustats.txt")
            do ii = 1, 2
              write(42, '(*(E16.7e3))'),  txzmean(ii), tyzmean(ii), tzzmean(ii), txzstd(ii), tyzstd(ii), tzzstd(ii)
            enddo
            close(42)

            print*, "start training."
          endif
        endif

        if (test_inference) then
          call MPI_ALLREDUCE(MPI_IN_PLACE,umean_inf,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,vmean_inf,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,wmean_inf,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,ustd_inf,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,vstd_inf,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
          call MPI_ALLREDUCE(MPI_IN_PLACE,wstd_inf,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)

          umean_inf = umean_inf / statcount
          vmean_inf = vmean_inf / statcount
          wmean_inf = wmean_inf / statcount

          ustd_inf = sqrt(ustd_inf / statcount - umean_inf**2)
          vstd_inf = sqrt(vstd_inf / statcount - vmean_inf**2)
          wstd_inf = sqrt(wstd_inf / statcount - wmean_inf**2)

          if (myid == 0) then
            print*, "test inference statistics computation complete"
            print*, "umean_inf", umean_inf
            print*, "vmean_inf", vmean_inf
            print*, "wmean_inf", wmean_inf
            print*, "ustd_inf", ustd_inf
            print*, "vstd_inf", vstd_inf
            print*, "wstd_inf", wstd_inf
            open(42, file="data/velstats_inf.txt")
            write(42, '(*(E16.7e3))'),  zc(kbot), umean_inf, vmean_inf, wmean_inf, ustd_inf, vstd_inf, wstd_inf
            close(42)
          endif
        endif

        !stop

        !is_training = .true.
        !timesample = 0
        !sampleidx = 0

      endif
    endif
    
    ! CHECKME: check if we want to do a training step:
    if (is_training .and. (timesample >= dtsample_scaled)) then
       timesample = 0.
       sampleidx = sampleidx + 1

       call cmpt_wallstresses_2d(n, is_bound, dzci, visc, u, v, p, &
            tauxz, tauyz, tauzz)

       !$acc kernels default(present)
       ! copy inputs
       input_local(1:n(1), 1:n(2), 1, sampleidx) = (tauxz(1:n(1),1:n(2),0) - txzmean(1)) / txzstd(1)
       input_local(1:n(1), 1:n(2), 2, sampleidx) = (tauyz(1:n(1),1:n(2),0) - tyzmean(1)) / tyzstd(1)
       input_local(1:n(1), 1:n(2), 3, sampleidx) = (tauzz(1:n(1),1:n(2),0) - tzzmean(1)) / tzzstd(1)
       ! copy labels
       ! CHECKME MODIFIED
       label_local(1:n(1), 1:n(2), 1, sampleidx) = (u(1:n(1),1:n(2), kbot) - umean(kbot)) / ustd(kbot)
       label_local(1:n(1), 1:n(2), 2, sampleidx) = (v(1:n(1),1:n(2), kbot) - vmean(kbot)) / vstd(kbot)
       label_local(1:n(1), 1:n(2), 3, sampleidx) = (w(1:n(1),1:n(2), kbot) - wmean(kbot)) / wstd(kbot)
       !$acc end kernels
       
       if (mod(sampleidx, trainbs * nranks) == 0) then
           sampleidx = 0
           call distribute_batches(input_local, input, n, ng)
           call distribute_batches(label_local, label, n, ng)

           ! DEBUG
           !write(fldnum,'(i7.7)') isteptrain
           !write(chkptnum,'(i4.4)') myid
           !filename = 'data/input_'//fldnum//'_'//chkptnum//'.h5'
           !!$acc update host(input, label)
           !call write_sample(input, label, filename)

           !write(fldnum,'(i7.7)') isteptrain
           !write(chkptnum,'(i4.4)') myid
           !filename = 'data/input_local_'//fldnum//'_'//chkptnum//'.h5'
           !!$acc update host(input_local, label_local)
           !call write_sample(input_local, label_local, filename)

           !call MPI_Barrier(MPI_COMM_WORLD, ierr)
           !stop(1)

           if (.not. is_validating) then
             isteptrain = isteptrain + 1
             !$acc host_data use_device(input, label)
             istat = torchfort_train(model_name, input, label, loss_value)
             !$acc end host_data
             if (istat /= TORCHFORT_RESULT_SUCCESS) stop

             if (myid == 0) print*, "Training step", isteptrain, "training loss = ", loss_value

             if (mod(isteptrain, nsamples_train / (trainbs * nranks)) == 0) then
               if (myid == 0) then
                 print*, "Completed train epoch", isteptrain * trainbs * nranks / nsamples_train
               endif
               is_validating = .true.
             endif

           else
             istepval = istepval + 1
             !$acc host_data use_device(input, output)
             istat = torchfort_inference(model_name, input, output)
             !$acc end host_data
             if (istat /= TORCHFORT_RESULT_SUCCESS) stop
             ! TODO: validation loss by component here
             !$acc kernels default(present)
             uerr = sum((label(1:ng(1),1:ng(2),1,:) - output(1:ng(1),1:ng(2),1,:))**2)/(trainbs * ng(1) * ng(2))
             verr = sum((label(1:ng(1),1:ng(2),2,:) - output(1:ng(1),1:ng(2),2,:))**2)/(trainbs * ng(1) * ng(2))
             werr = sum((label(1:ng(1),1:ng(2),3,:) - output(1:ng(1),1:ng(2),3,:))**2)/(trainbs * ng(1) * ng(2))
             !$acc end kernels

             call MPI_ALLREDUCE(MPI_IN_PLACE,uerr,1,MPI_REAL_RP,MPI_SUM,MPI_COMM_WORLD,ierr)
             call MPI_ALLREDUCE(MPI_IN_PLACE,verr,1,MPI_REAL_RP,MPI_SUM,MPI_COMM_WORLD,ierr)
             call MPI_ALLREDUCE(MPI_IN_PLACE,werr,1,MPI_REAL_RP,MPI_SUM,MPI_COMM_WORLD,ierr)
             uerr = uerr / nranks
             verr = verr / nranks
             werr = werr / nranks

             if (myid == 0) then
               print*, "Validate step", istepval
               print*, "u_err", uerr, "v_err", verr, "w_err", werr

               istat = torchfort_wandb_log(model_name, "u_val_err", istepval, uerr)
               if (istat /= TORCHFORT_RESULT_SUCCESS) stop
               istat = torchfort_wandb_log(model_name, "v_val_err", istepval, verr)
               if (istat /= TORCHFORT_RESULT_SUCCESS) stop
               istat = torchfort_wandb_log(model_name, "w_val_err", istepval, werr)
               if (istat /= TORCHFORT_RESULT_SUCCESS) stop
             endif

             if (mod(istepval, nsamples_val / (trainbs * nranks)) == 0) then
               write(fldnum,'(i7.7)') isteptrain
               write(chkptnum,'(i4.4)') myid
               filename = 'data/sample_'//fldnum//'_'//chkptnum//'.h5'
               !$acc update host(output, label)
               call write_sample(output, label, filename)
               print "(a34,i10,a12,a100)", "Writing validation sample at step ", istepval, " filename = ", filename
               print*, "Completed validation epoch", istepval * trainbs * nranks / nsamples_val
               is_validating = .false.

               if (myid == 0) then
                 write(trainckptnum,'(i7.7)') isteptrain
                 filename = 'torchfort_checkpoint_'//trainckptnum
                 print "(a20,i10,a12,a100)", "Writing checkpoint ", isteptrain, " directory = ", filename
                 istat = torchfort_save_checkpoint(model_name, filename)
                 if (istat /= TORCHFORT_RESULT_SUCCESS) stop
               endif

               ! Save CaNS checkpoint
               filename = 'fld.bin'
               if (myid == 0) print*, "Writing CaNS checkpoint..."
               !$acc update self(u,v,w,p)
               call load_all('w',trim(datadir)//trim(filename),MPI_COMM_WORLD,ng,[1,1,1],lo,hi,u,v,w,p,time,istep)

             endif
           endif
       endif

    end if

    !
    ! check simulation stopping criteria
    !
    if(stop_type(1)) then ! maximum number of time steps reached
      if(istep >= nstep   ) is_done = is_done.or..true.
    end if
    if(stop_type(2)) then ! maximum simulation time reached
      if(time  >= time_max) is_done = is_done.or..true.
    end if
    if(stop_type(3)) then ! maximum wall-clock time reached
      tw = (MPI_WTIME()-twi)/3600.
      if(tw    >= tw_max  ) is_done = is_done.or..true.
    end if
    if(mod(istep,icheck) == 0) then
      if(myid == 0) print*, 'Checking stability and divergence...'
      call chkdt(n,dl,dzci,dzfi,visc,u,v,w,dtmax)
      dt  = min(cfl*dtmax,dtmin)
      if(myid == 0) print*, 'dtmax = ', dtmax, 'dt = ', dt
      if(dtmax < small) then
        if(myid == 0) print*, 'ERROR: time step is too small.'
        if(myid == 0) print*, 'Aborting...'
        is_done = .true.
        kill = .true.
      end if
      dti = 1./dt
      call chkdiv(lo,hi,dli,dzfi,u,v,w,divtot,divmax)
      if(myid == 0) print*, 'Total divergence = ', divtot, '| Maximum divergence = ', divmax
#if !defined(_MASK_DIVERGENCE_CHECK)
      if(divmax > small.or.is_nan(divtot)) then
        if(myid == 0) print*, 'ERROR: maximum divergence is too large.'
        if(myid == 0) print*, 'Aborting...'
        is_done = .true.
        kill = .true.
      end if
#endif
    end if
    !
    ! output routines below
    !
    if(mod(istep,iout0d) == 0) then
      !allocate(var(4))
      var(1) = 1.*istep
      var(2) = dt
      var(3) = time
      call out0d(trim(datadir)//'time.out',3,var)
      !
      if(any(is_forced(:)).or.any(abs(bforce(:)) > 0.)) then
        meanvelu = 0.
        meanvelv = 0.
        meanvelw = 0.
        if(is_forced(1).or.abs(bforce(1)) > 0.) then
          call bulk_mean(n,grid_vol_ratio_f,u,meanvelu)
        end if
        if(is_forced(2).or.abs(bforce(2)) > 0.) then
          call bulk_mean(n,grid_vol_ratio_f,v,meanvelv)
        end if
        if(is_forced(3).or.abs(bforce(3)) > 0.) then
          call bulk_mean(n,grid_vol_ratio_c,w,meanvelw)
        end if
        if(.not.any(is_forced(:))) dpdl(:) = -bforce(:) ! constant pressure gradient
        var(1)   = time
        var(2:4) = dpdl(1:3)
        var(5:7) = [meanvelu,meanvelv,meanvelw]
        call out0d(trim(datadir)//'forcing.out',7,var)
      end if
    end if
    write(fldnum,'(i7.7)') istep
    if(mod(istep,iout1d) == 0) then
      !$acc wait
      !$acc update self(u,v,w,p)
      include 'out1d.h90'
    end if
    if(mod(istep,iout2d) == 0) then
      !$acc wait
      !$acc update self(u,v,w,p)
      include 'out2d.h90'
    end if
    if(mod(istep,iout3d) == 0) then
      !$acc wait
      !$acc update self(u,v,w,p)
      include 'out3d.h90'
    end if
    if(mod(istep,isave ) == 0.or.(is_done.and..not.kill)) then
      if(is_overwrite_save) then
        filename = 'fld.bin'
      else
        filename = 'fld_'//fldnum//'.bin'
        if(nsaves_max > 0) then
          if(savecounter >= nsaves_max) savecounter = 0
          savecounter = savecounter + 1
          write(chkptnum,'(i4.4)') savecounter
          filename = 'fld_'//chkptnum//'.bin'
          var(1) = 1.*istep
          var(2) = time
          var(3) = 1.*savecounter
          call out0d(trim(datadir)//'log_checkpoints.out',3,var)
        end if
      end if
      !$acc wait
      !$acc update self(u,v,w,p)
      call load_all('w',trim(datadir)//trim(filename),MPI_COMM_WORLD,ng,[1,1,1],lo,hi,u,v,w,p,time,istep)
      if(.not.is_overwrite_save) then
        !
        ! fld.bin -> last checkpoint file (symbolic link)
        !
        call gen_alias(myid,trim(datadir),trim(filename),'fld.bin')
      end if
      if(myid == 0) print*, '*** Checkpoint saved at time = ', time, 'time step = ', istep, '. ***'
    end if
#if defined(_TIMING)
      !$acc wait(1)
      dt12 = MPI_WTIME()-dt12
      call MPI_ALLREDUCE(dt12,dt12av ,1,MPI_REAL_RP,MPI_SUM,MPI_COMM_WORLD,ierr)
      call MPI_ALLREDUCE(dt12,dt12min,1,MPI_REAL_RP,MPI_MIN,MPI_COMM_WORLD,ierr)
      call MPI_ALLREDUCE(dt12,dt12max,1,MPI_REAL_RP,MPI_MAX,MPI_COMM_WORLD,ierr)
      if(myid == 0) print*, 'Avrg, min & max elapsed time: '
      if(myid == 0) print*, dt12av/(1.*product(dims)),dt12min,dt12max
#endif
  end do
  !
  ! clear ffts
  !
  call fftend(arrplanp)
#if defined(_IMPDIFF) && !defined(_IMPDIFF_1D)
  call fftend(arrplanu)
  call fftend(arrplanv)
  call fftend(arrplanw)
#endif
  if(myid == 0.and.(.not.kill)) print*, '*** Fim ***'
  call decomp_2d_finalize
  call MPI_FINALIZE(ierr)
end program cans
