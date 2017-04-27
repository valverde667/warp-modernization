# Main class written by J.-L. Vay at Lawrence Berkeley National Laboratory
# Lab snapshot diagnostics written in the research group of A. R. Maier at the University of Hamburg
# with contributions from I. Dornmair, V. Hanus, S. Jalas, M. Kirchen

from warp import *
import string
try:
  import h5py
  l_h5py=1 
except:
  l_h5py=0

class Boosted_Frame(object):
  """
Class transforming particle distribution from lab frame to boosted frame. 
Boosted particles can optionally be injected through a plane. In this case, the 
boosted particles are moved from the main top.pgroup to a separate particle group, 
drift at velocity top.vbeam until they reach the injection plane though which 
they are injected using Warp's injection routines.
In the current implementation, the distribution needs to fit entirely in the 
simulation zone, which is too restrictive for some applications and will need 
to be lifted in the future.
  """
  def __init__(self,gammaframe,direction=1.,l_setselfb=1):
    top.boost_gamma=gammaframe
    self.gammaframe=gammaframe
    self.l_setselfb=l_setselfb
    self.betaframe  = direction*sqrt(1.-1./self.gammaframe**2)
    self.betabeam_lab=top.vbeam/clight
    self.betabeamfrm_lab=top.vbeamfrm/clight
    top.vbeam_lab = top.vbeam
#    top.gammabar_lab = top.gammabar
    top.gammabar_lab=1./sqrt(1.-(top.vbeam_lab/clight)**2)
    top.vbeam=clight*(self.betabeam_lab-self.betaframe)/(1.-self.betabeam_lab*self.betaframe)
    top.vbeamfrm=clight*(self.betabeamfrm_lab-self.betaframe)/(1.-self.betabeamfrm_lab*self.betaframe)
    top.gammabar=1./sqrt(1.-(top.vbeam/clight)**2)
    # --- defines lists for particle groups to inject
    self.pgroups  = []
    self.zinjects = []
    self.vbeams   = []
    self.list_species = []
    # --- sets selfb arrays
    if self.l_setselfb:
      fselfbcopy = top.fselfb.copy()
      top.fselfb[...]=(top.fselfb[...]-self.betaframe*clight)/(1.-top.fselfb[...]*self.betaframe/clight)
      for js in range(shape(top.pgroup.fselfb)[0]):
        for jss in range(top.nsselfb):
          if top.pgroup.fselfb[js] == fselfbcopy[jss]:
            top.pgroup.fselfb[js]=top.fselfb[jss]

  def boost(self,species,zinject=0.,tinit=0.,l_inject_plane=1,lallindomain=0,
            l_rmzmean=1.,l_deprho=1,l_savezinit=0,l_focus=1,l_project=1):
   print 'enter boost',top.pgroup.nps
   if l_savezinit:
     if top.zbirthlabpid==0:
       top.zbirthlabpid = nextpid()
       top.pgroup.npid = top.npid
       top.pgroup.gchange()
     iupr=-1
     for js in species.jslist:
       ilpr=iupr+1
       iupr=ilpr+getn(js=js,bcast=0,gather=0)
       if getn(js=js,bcast=0,gather=0):
         top.pgroup.pid[ilpr:iupr,top.zbirthlabpid-1] = top.pgroup.zp[ilpr:iupr].copy()
   if l_inject_plane:
    pg = top.pgroup
    self.list_species+=[species]
    self.zinjects+=[zinject]
    self.tinit=tinit
    self.l_rmzmean=l_rmzmean
    self.pgroups.append(ParticleGroup())
    self.ipgrp = -1
    self.pgroup = self.pgroups[-1]
#    self.pgroup.ns = pg.ns#len(species.jslist)
    self.pgroup.ns = len(species.jslist)
    self.pgroup.npmax = species.getn(bcast=0,gather=0)
    self.pgroup.npid = pg.npid
    self.pgroup.lebcancel_pusher = top.pgroup.lebcancel_pusher
    self.pgroup.gchange()
    iupr=-1
    for jspr,js in enumerate(species.jslist):
      ilpr=iupr+1
      iupr=ilpr+getn(pgroup=pg,js=js,bcast=0,gather=0)
      self.pgroup.sq[jspr] = pg.sq[js]
      self.pgroup.sm[jspr] = pg.sm[js]
      self.pgroup.sw[jspr] = pg.sw[js]
      self.pgroup.sid[jspr] = pg.sid[js]
      self.pgroup.ndts[jspr] = pg.ndts[js]
      self.pgroup.ldts[jspr] = pg.ldts[js]
      self.pgroup.lvdts[jspr] = pg.lvdts[js]
      self.pgroup.iselfb[jspr] = pg.iselfb[js]
      self.pgroup.dtscale[jspr] = pg.dtscale[js]
      self.pgroup.limplicit[jspr] = pg.limplicit[js]
      self.pgroup.iimplicit[jspr] = pg.iimplicit[js]
      self.pgroup.zshift[jspr] = pg.zshift[js]
      self.pgroup.ins[jspr]=ilpr+1
      self.pgroup.nps[jspr]=getn(pgroup=pg,js=js,bcast=0,gather=0)
      if l_inject_plane:
       if getn(pgroup=pg,js=js,bcast=0,gather=0)>0: 
        z=getz(pgroup=pg,js=js,bcast=0,gather=0)
       else:
        z=array([])
       if self.l_rmzmean:
         zmean=globalave(z)
       else:
         zmean=0.
       vz = getvz(pgroup=pg,js=js,bcast=0,gather=0)
       self.betabeam_lab = globalave(vz)/clight
       self.vbeams.append(clight*(self.betabeam_lab-self.betaframe)/(1.-self.betabeam_lab*self.betaframe))
       if getn(pgroup=pg,js=js,bcast=0,gather=0)>0: 
        gaminvbeam_lab = getgaminv(pgroup=pg,js=js,bcast=0,gather=0)
        betabeam_lab  = sqrt(1.-gaminvbeam_lab*gaminvbeam_lab)
        betabeam_frame = (betabeam_lab-self.betaframe)/(1.-betabeam_lab*self.betaframe)
        gammabeam_frame  = 1./sqrt(1.-betabeam_frame*betabeam_frame)
        zcopy = z.copy()
        z=z-zmean
        # --- get data at z=0
        vx = getvx(pgroup=pg,js=js,bcast=0,gather=0)
        vy = getvy(pgroup=pg,js=js,bcast=0,gather=0)
        vz = getvz(pgroup=pg,js=js,bcast=0,gather=0)

        t = z/vz
        x = getx(pgroup=pg,js=js,bcast=0,gather=0)
        y = gety(pgroup=pg,js=js,bcast=0,gather=0)
        if not l_project:
          x-=t*vx
          y-=t*vy
        # --- correct for focusing effect from shift from z=0 to zinject
        if l_focus:
          tfoc = -zinject*self.gammaframe/vz#pr
          x = x-tfoc*vx#pr
          y = y-tfoc*vy#pr
        # --- get data in boosted frame
        tpr = -self.gammaframe*t
        zpr = self.gammaframe*self.betaframe*clight*t
       else:
        zpr=array([])
       if top.boost_z0==0.:
         top.boost_z0 = -globalmax(zpr)
       if getn(pgroup=pg,js=js,bcast=0,gather=0)>0: 
        fact = 1./(1.-self.betaframe*vz/clight)
        vxpr = vx*fact/self.gammaframe
        vypr = vy*fact/self.gammaframe
        vzpr = (vz-self.betaframe*clight)*fact
       # --- get data at t=0 in boosted frame
#        zpr = zpr - vzpr*tpr
        zpr = zpr - self.vbeams[-1]*tpr
#        zpr = zcopy*top.gammabar_lab/top.gammabar
        # --- make sure that z<=0
#        zpr += top.boost_z0 
        # --- sets location of beam center at t=0 in boosted frame
        gammapr = 1./sqrt(1.-(vxpr*vxpr+vypr*vypr+vzpr*vzpr)/clight**2)
        self.pgroup.uxp[ilpr:iupr]=vxpr*gammapr
        self.pgroup.uyp[ilpr:iupr]=vypr*gammapr
        self.pgroup.uzp[ilpr:iupr]=vzpr*gammapr
        self.pgroup.gaminv[ilpr:iupr]=1./gammapr
        self.pgroup.xp[ilpr:iupr] = x
        self.pgroup.yp[ilpr:iupr] = y
        self.pgroup.zp[ilpr:iupr] = zpr
        if pg.npid>0:self.pgroup.pid[ilpr:iupr,:] = getpid(pgroup=pg,js=js,bcast=0,gather=0,id=-1)
        if top.uxoldpid>0:self.pgroup.pid[ilpr:iupr,top.uxoldpid-1]=self.pgroup.uxp[ilpr:iupr]
        if top.uyoldpid>0:self.pgroup.pid[ilpr:iupr,top.uyoldpid-1]=self.pgroup.uyp[ilpr:iupr]
        if top.uzoldpid>0:self.pgroup.pid[ilpr:iupr,top.uzoldpid-1]=self.pgroup.uzp[ilpr:iupr]
      pg.nps[js]=0
#      if pg.fselfb[js] != 0.:
#        pg.fselfb[js]=(pg.fselfb[js]-self.betaframe*clight)/(1.-pg.fselfb[js]*self.betaframe/clight)
      self.pgroup.fselfb[jspr] = pg.fselfb[js]
    # --- check for particle out of bounds and exchange particles among processors if needed
    top.ns=self.pgroup.ns
#    zpartbnd(self.pgroup,w3d.zmmax,w3d.zmmin,w3d.dz)
    particlegridboundaries3d(top.pgroup,-1)
    top.ns=top.pgroup.ns
    # --- Specify injection of the particles
    top.inject   = 1 #3
    top.injctspc = 1
    top.npinject = 0
    top.zinject  = zinject
    top.ainject  = w3d.xmmax
    top.binject  = w3d.ymmax
    top.apinject = 0.e0
    top.bpinject = 0.e0
    top.lvinject = false  # if false, source conductor input by user
    top.inj_d    = 2.0
    top.inj_f    = 1.0
    top.finject[0][1:]=0.
    top.linj_efromgrid=True
    vbeamfrmtmp = top.vbeamfrm
    injctint(pg)
    top.vbeamfrm = vbeamfrmtmp
    w3d.l_inj_user_particles = true
    w3d.l_inj_user_particles_v = true
    w3d.l_inj_user_particles_dt = true
    w3d.l_inj_zmminmmaxglobal = true
#    installuserparticlesinjection(self.add_boosted_species)
    if len(self.pgroups)==1:installbeforestep(self.add_boosted_species_multigroups)
    if l_deprho:
      self.depos=top.depos.copy()
      top.depos='none'
      installbeforefs(self.add_boosted_rho)
    self.hn = []
    self.hinj = []
    self.hbf = []
   else:
    pg=top.pgroup
    for jspr,js in enumerate(species.jslist):
#      if pg.fselfb[js] != 0.:
#        pg.fselfb[js]=(pg.fselfb[js]-self.betaframe*clight)/(1.-pg.fselfb[js]*self.betaframe/clight)
      il=top.pgroup.ins[js]-1
      iu=il+top.pgroup.nps[js]
      if getn(pgroup=pg,js=js,bcast=0,gather=0)>0: 
        z=getz(pgroup=pg,js=js,bcast=0,gather=0)
      else:
        z=0.
      zmean=globalave(z)
      if getn(pgroup=pg,js=js,bcast=0,gather=0)>0: 
        uzfrm = self.gammaframe*self.betaframe*clight
        tpr =  self.gammaframe*top.time-uzfrm*top.pgroup.zp[il:iu]/clight**2
        top.pgroup.zp[il:iu] = self.gammaframe*top.pgroup.zp[il:iu]-uzfrm*top.time
#        top.pgroup.zp[il:iu]=zmean+(top.pgroup.zp[il:iu]-zmean)/(self.gammaframe*(1.-self.betaframe*self.betabeam_lab))
        vx = getvx(pgroup=pg,js=js,bcast=0,gather=0)
        vy = getvy(pgroup=pg,js=js,bcast=0,gather=0)
        vz = getvz(pgroup=pg,js=js,bcast=0,gather=0)
        fact = 1./(1.-self.betaframe*vz/clight)
        vxpr = vx*fact/self.gammaframe
        vypr = vy*fact/self.gammaframe
        vzpr = (vz-self.betaframe*clight)*fact
        top.pgroup.xp[il:iu] = top.pgroup.xp[il:iu] - tpr*vxpr
        top.pgroup.yp[il:iu] = top.pgroup.yp[il:iu] - tpr*vypr
        top.pgroup.zp[il:iu] = top.pgroup.zp[il:iu] - tpr*vzpr
        gammapr = 1./sqrt(1.-(vxpr*vxpr+vypr*vypr+vzpr*vzpr)/clight**2)
        top.pgroup.uxp[il:iu]=vxpr*gammapr
        top.pgroup.uyp[il:iu]=vypr*gammapr
        top.pgroup.uzp[il:iu]=vzpr*gammapr
        top.pgroup.gaminv[il:iu]=1./gammapr
        if top.uxoldpid>0:top.pgroup.pid[il:iu,top.uxoldpid-1]=top.pgroup.uxp[il:iu]
        if top.uyoldpid>0:top.pgroup.pid[il:iu,top.uyoldpid-1]=top.pgroup.uyp[il:iu]
        if top.uzoldpid>0:top.pgroup.pid[il:iu,top.uzoldpid-1]=top.pgroup.uzp[il:iu]
   if not lallindomain:particleboundaries3d(top.pgroup,-1,False)
   print 'exit boost',top.pgroup.nps
   
  def add_boosted_species_multigroups(self):
      do_inject = 0
      w3d.npgrp = 0
      for self.ipgrp,self.pgroup in enumerate(self.pgroups):
          self.vbeam = self.vbeams[self.ipgrp]
          self.species = self.list_species[self.ipgrp]
          # --- check whether pid arrays need to be reshaped 
          if self.pgroup.npid != top.pgroup.npid:
              self.pgroup.npid = top.pgroup.npid
              self.pgroup.gchange()
          # --- push longitudinal particle positions
          for js in range(self.pgroup.ns):
              if self.pgroup.nps[js]>0:
                  il=self.pgroup.ins[js]-1
                  iu=il+self.pgroup.nps[js]
                  if top.zoldpid>0:self.pgroup.pid[il:iu,top.zoldpid-1]=self.pgroup.zp[il:iu].copy()
                  self.pgroup.zp[il:iu]+=top.dt*self.vbeam
          # --- does injection for each particle group
          if any(parallelsum(self.pgroup.nps)>0):
              do_inject = 1
              top.inject=1
              self.add_boosted_species()
          if not do_inject:
              w3d.npgrp = 0
              gchange("Setpwork3d")
              top.inject=0
#        uninstallbeforestep(self.add_boosted_species)
                    
  def add_boosted_species(self):
    nps = parallelsum(self.pgroup.nps)
#    top.finject[0][1:]=0.
#    if all(nps==0):
#      top.inject=0
    self.zinject = self.zinjects[self.ipgrp]
    top.zinject=self.zinject-top.time*self.betaframe*clight
    for js in range(self.pgroup.ns):
     if self.pgroup.nps[js]>0:
      il=self.pgroup.ins[js]-1
      iu=il+self.pgroup.nps[js]
      ii=compress(self.pgroup.zp[il:iu]>top.zinject,il+arange(getn(pgroup=self.pgroup,js=js,bcast=0,gather=0)))
      if len(ii)>0:
        w3d.npgrp = len(ii)
        w3d.npidgrp = top.npid
        gchange("Setpwork3d")
        top.finject[0][:]=0.
        top.finject[0][self.species.jslist[0]]=1.
        gi=take(self.pgroup.gaminv,ii)
        vz = take(self.pgroup.uzp,ii)*gi
        w3d.xt = take(self.pgroup.xp,ii).copy()
        w3d.yt = take(self.pgroup.yp,ii).copy()
        w3d.uxt = take(self.pgroup.uxp,ii)*gi
        w3d.uyt = take(self.pgroup.uyp,ii)*gi
        w3d.uzt = vz
#        w3d.bpt = (take(self.pgroup.zp,ii)-top.zinject)/vz
        w3d.bpt = (take(self.pgroup.zp,ii)-top.zinject)/self.vbeam
        for ipid in range(top.npid):
          w3d.pidt[:,ipid] = take(self.pgroup.pid[:,ipid],ii)
#        gi=getgaminv(pgroup=self.pgroup,js=js,bcast=0,gather=0)
        put(self.pgroup.gaminv,ii,0.)     
        npo = self.pgroup.nps[0]
        processlostpart(self.pgroup,js+1,top.clearlostpart,top.time+top.dt*self.pgroup.ndts[js],top.zbeam)
    self.hn.append(getn())
    self.hinj.append(globalsum(w3d.npgrp))
    self.hbf.append(globalsum(self.pgroup.nps[0]))
        
  def pln(self):
    pla(self.hn)
    pla(self.hinj,color=red)
    pla(self.hbf,color=blue)
    pla(cumsum(self.hinj),color=green)

  def add_boosted_speciesold(self):
    for js in range(self.pgroup.ns):
     if self.pgroup.nps[js]>0:
      il=self.pgroup.ins[js]-1
      iu=il+self.pgroup.nps[js]
#      self.pgroup.xp[il:iu]+=top.dt*getvx(pgroup=self.pgroup,js=js,bcast=0,gather=0)
#      self.pgroup.yp[il:iu]+=top.dt*getvy(pgroup=self.pgroup,js=js,bcast=0,gather=0)
#      self.pgroup.zp[il:iu]+=top.dt*getvz(pgroup=self.pgroup,js=js,bcast=0,gather=0) # WARNING: this can cause particles to get out of bounds
      self.pgroup.zp[il:iu]+=top.dt*top.vbeam
    for js in range(self.pgroup.ns):
     if self.pgroup.nps[js]>0:
      il=self.pgroup.ins[js]-1
      iu=il+self.pgroup.nps[js]
      ii=compress(self.pgroup.zp[il:iu]>self.zinject-top.time*self.betaframe*clight,il+arange(getn(pgroup=self.pgroup,js=js,bcast=0,gather=0)))
      if len(ii)>0:
        if self.pgroup.npid>0:
          pid=take(self.pgroup.pid,ii,0)
        else:
          pid=0.
        self.species.addpart(x=take(self.pgroup.xp,ii),
                           y=take(self.pgroup.yp,ii),
                           z=take(self.pgroup.zp,ii),
                           vx=take(self.pgroup.uxp,ii),
                           vy=take(self.pgroup.uyp,ii),
                           vz=take(self.pgroup.uzp,ii),
                           gi=take(self.pgroup.gaminv,ii),
                           pid=pid,
                           lmomentum=true,lallindomain=true)
        gi=getgaminv(pgroup=self.pgroup,js=js,bcast=0,gather=0)
        put(self.pgroup.gaminv,ii,0.)
        processlostpart(self.pgroup,js+1,top.clearlostpart,top.time+top.dt*self.pgroup.ndts[js],top.zbeam)

  def add_boosted_rho(self):
    for self.pgroup in self.pgroups:
        if self.pgroup.npid != top.pgroup.npid:
            self.pgroup.npid = top.pgroup.npid
            self.pgroup.gchange()
#    if string.rstrip(top.depos.tostring())=='none': return
#    w3d.lbeforelr=0
    fs=getregisteredsolver()
    top.depos=self.depos
    doit = True
    for js in range(self.pgroup.ns):
        if self.pgroup.nps[js]>0:
            il=self.pgroup.ins[js]-1
            iu=il+self.pgroup.nps[js]
            doit = doit and (minnd(self.pgroup.xp[il:iu])>w3d.xmminlocal) and \
               (maxnd(self.pgroup.xp[il:iu])<w3d.xmmaxlocal) and \
               (minnd(self.pgroup.yp[il:iu])>w3d.ymminlocal) and \
               (maxnd(self.pgroup.yp[il:iu])<w3d.ymmaxlocal) and \
               (minnd(self.pgroup.zp[il:iu])>top.zgrid+w3d.zmminlocal) and \
               (maxnd(self.pgroup.zp[il:iu])<top.zgrid+w3d.zmmaxlocal)
    if doit:
       fs.loadrho(pgroups=self.pgroups+[top.pgroup])
    else:
       fs.loadrho(pgroups=[top.pgroup])
    top.depos='none'
#    w3d.lbeforelr=1

  def add_boosted_rho_old(self):
    if rstrip(top.depos.tostring())=='none': return
    w3d.lbeforelr=0
    if 1:#getn(pgroup=self.pgroup)>0:
      fs=getregisteredsolver()
      pg=top.pgroup
#      fs.zerosourcep()
#      top.laccumulate_rho=true
      top.depos=self.depos
      fs.loadrho(pgroups=[top.pgroup,self.pgroup])
#      top.laccumulate_rho=false
      self.depos=top.depos.copy()
      top.depos='none'
      fs.aftersetsourcep()
#      fs.aftersetsourcep(lzero=1)
    w3d.lbeforelr=1

  def getn(self):
    n1 = self.species.getn()
    n2 = getn(pgroup=self.pgroup)
    return n1+n2
    
  def getx(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.getx(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = getx(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])
    
  def gety(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.gety(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = gety(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])
    
  def getz(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.getz(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = getz(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])
    
  def getgaminv(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.getgaminv(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = getgaminv(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])
        
  def getux(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.getux(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = getux(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])
    
  def getuy(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.getuy(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = getuy(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])
    
  def getuz(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.getuz(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = getuz(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])
    
  def getvx(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.getvx(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = getvx(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])
    
  def getvy(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.getvy(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = getvy(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])
    
  def getvz(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.getvz(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = getvz(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])
    
  def getke(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.getke(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = getke(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])
    
  def getpid(self,**kw):
    n1 = self.species.getn(**kw)
    if n1>0:
      z1 = self.species.getpid(**kw)
    else:
      z1 = array([])
    n2 = getn(pgroup=self.pgroup,**kw)
    if n2>0:
      z2 = getpid(pgroup=self.pgroup,**kw)
    else:
      z2 = array([])
    return concatenate([z1,z2])

  def dump(self,filename='pdump.pdb'):
        if self.getn()==0:
            return
            x=y=z=ux=uy=uz=gi=pidNone
        else:
            x=self.getx()
            y=self.gety()
            z=self.getz()
            ux=self.getux()
            uy=self.getuy()
            uz=self.getuz()
            gi=self.getgaminv()
            if top.npid>0:
                pid=self.getpid()
            else:
                pid=None
        if me==0:
            import PWpickle as PW
            f=PW.PW(filename)
            f.time=top.time
            f.x=x
            f.y=y
            f.z=z
            f.ux=ux
            f.uy=uy
            f.uz=uz
            f.gi=gi
            f.pid=pid
            f.close()

  def get_density(self,xmin=None,xmax=None,nx=None,ymin=None,ymax=None,ny=None,zmin=None,zmax=None,
                    nz=None,lost=0,charge=0,dens=None,l_minmax_grid=true,l_dividebyvolume=1,l4symtry=None,l2symtry=None):
        if l_minmax_grid:
            if xmin is None:xmin=w3d.xmmin
            if xmax is None:xmax=w3d.xmmax
            if ymin is None:ymin=w3d.ymmin
            if ymax is None:ymax=w3d.ymmax
            if zmin is None:zmin=w3d.zmmin+top.zgrid
            if zmax is None:zmax=w3d.zmmax+top.zgrid
            if l4symtry is None:l4symtry=w3d.l4symtry
            if l2symtry is None:l2symtry=w3d.l2symtry
        else:
            if xmin is None:xmin=min(self.getx())
            if xmax is None:xmax=max(self.getx())
            if ymin is None:ymin=min(self.gety())
            if ymax is None:ymax=max(self.gety())
            if zmin is None:zmin=min(self.getz())
            if zmax is None:zmax=max(self.getz())
            if l4symtry is None:l4symtry=false
            if l2symtry is None:l2symtry=false
        if dens is None:
            if nx is None:nx=w3d.nx
            if ny is None:ny=w3d.ny
            if nz is None:nz=w3d.nz
            if w3d.solvergeom is w3d.XYgeom:
                density = fzeros([nx+1,ny+1],'d')
                densityc = fzeros([nx+1,ny+1],'d')
            else:
                if w3d.solvergeom in [w3d.Zgeom]:
                    density = fzeros([nz+1],'d')
                    densityc = fzeros([nz+1],'d')
                elif w3d.solvergeom in [w3d.XZgeom,w3d.RZgeom]:
                    density = fzeros([nx+1,nz+1],'d')
                    densityc = fzeros([nx+1,nz+1],'d')
                else:
                    density = fzeros([nx+1,ny+1,nz+1],'d')
                    densityc = fzeros([nx+1,ny+1,nz+1],'d')
        else:
            if w3d.solvergeom is w3d.XYgeom:
                nx = shape(dens)[0]-1
                ny = shape(dens)[1]-1
            else:
                if w3d.solvergeom in [w3d.Zgeom]:
                    nz = shape(dens)[0]-1
                if w3d.solvergeom in [w3d.XZgeom,w3d.RZgeom]:
                    nx = shape(dens)[0]-1
                    nz = shape(dens)[1]-1
                else:
                    nx = shape(dens)[0]-1
                    ny = shape(dens)[1]-1
                    nz = shape(dens)[2]-1
            density = dens
            densityc = 0.*dens

        np=0
        for pgroup in [top.pgroup,self.pgroup]:
          for js in self.species.jslist:
            np+=getn(pgroup=pgroup,js=js)
        if np == 0:
            if dens is None:
                return density
            else:
                return
        for pgroup in [top.pgroup,self.pgroup]:
          for js in self.species.jslist:
            x=getx(pgroup=pgroup,js=js,lost=lost,gather=0)
            y=gety(pgroup=pgroup,js=js,lost=lost,gather=0)
            z=getz(pgroup=pgroup,js=js,lost=lost,gather=0)
            if w3d.solvergeom==w3d.RZgeom:x=sqrt(x*x+y*y)
            np=shape(x)[0]
            if np > 0:
                if top.wpid == 0:
                    w=self.pgroup.sw[js]*ones(np,'d')
                else:
                    w=self.pgroup.sw[js]*getpid(pgroup=pgroup,js=js,id=top.wpid-1,gather=0)
                if charge:w*=self.pgroup.sq[js]
                if w3d.solvergeom is w3d.Zgeom:
                    deposgrid1d(1,np,z,w,nz,density,densityc,zmin,zmax)
                elif w3d.solvergeom is w3d.XYgeom:
                    deposgrid2d(1,np,x,y,w,nx,ny,density,densityc,xmin,xmax,ymin,ymax)
                elif w3d.solvergeom in [w3d.XZgeom,w3d.RZgeom]:
                    deposgrid2d(1,np,x,z,w,nx,nz,density,densityc,xmin,xmax,zmin,zmax)
                else:
                    deposgrid3d(1,np,x,y,z,w,nx,ny,nz,density,densityc,xmin,xmax,ymin,ymax,zmin,zmax)
        if w3d.solvergeom is w3d.Zgeom:
            if l_dividebyvolume:
                density*=nz/(zmax-zmin)
        elif w3d.solvergeom is w3d.XYgeom:
            if l_dividebyvolume:
                density*=nx*ny/((xmax-xmin)*(ymax-ymin))
                if l4symtry:
                    density[0,:] *= 2
                    density[:,0] *= 2
                if l2symtry:
                    density[:,0] *= 2
                if w3d.boundxy is periodic:
                    density[0,:] += density[-1,:]; density[-1,:]=density[0,:]
                    density[:,0] += density[:,-1]; density[:,-1]=density[:,0]
        else:
            if w3d.solvergeom in [w3d.XZgeom,w3d.RZgeom]:
                if l_dividebyvolume:
                    density*=nx*nz/((xmax-xmin)*(zmax-zmin))
                    if l4symtry:
                        density[0,:] *= 2
                    if w3d.boundxy is periodic:
                        density[0,:] += density[-1,:]; density[-1,:]=density[0,:]
                    if w3d.bound0 is periodic:
                        density[:,0] += density[:,-1]; density[:,-1]=density[:,0]
                    if w3d.solvergeom==w3d.RZgeom:
                        dr = (xmax-xmin)/nx
                        r = arange(nx+1)*dr
                        for j in range(1,nx+1):
                            density[j,:] /= 2.*pi*r[j]
                        density[0,:] /= pi*dr/2
            else:
                if l_dividebyvolume:
                    density*=nx*ny*nz/((xmax-xmin)*(ymax-ymin)*(zmax-zmin))
                    if l4symtry:
                        density[0,:,:] *= 2
                        density[:,0,:] *= 2
                    if l2symtry:
                        density[:,0,:] *= 2
                    if w3d.boundxy is periodic:
                        density[0,:,:] += density[-1,:,:]; density[-1,:,:]=density[0,:,:]
                        density[:,0,:] += density[:,-1,:]; density[:,-1,:]=density[:,0,:]
                    if w3d.bound0 is periodic:
                        density[:,:,0] += density[:,:,-1]; density[:,:,-1]=density[:,:,0]
        density[...] = parallelsum(density)
        if dens is None: return density

  def create_lab_snapshots(self,n_lab_snapshots, 
                                datatypes, 
                                l_restart=False, 
                                output_dir='data',
                                field_snapshot_divider = 1,
                                elec_snapshot_divider  = 1,
                                downsample_rate_lab_x  = 1,
                                downsample_rate_lab_y  = 1,
                                downsample_rate_lab_z  = 1,
                                momentum_resolution    = 400,
                                momentum_threshold_low = 5,
                                momentum_threshold_high= 100,
                                gamma_threshold_lab    = 1.,
                                l_transverse_profile = 0, 
                                plasma_long_profile = None,
                                plasma_trans_profile = None,
                                window_velocity_lab  = 0.,
                                zmin_window_lab = 0.,
                                zmax_window_lab = 0.,
                                elec = None,
                                beam = None,
                                em   = None,
                                density = 0.,
                                l_beam = False,
                                runid = "",
                                l_collapse=True,
                                ):
      """Creates files for backtransformed lab frame data output. Files are created for each datatype (elec, field, beam)."""
      self.datatypes = datatypes
      self.output_dir = output_dir
      self.n_lab_snapshots = n_lab_snapshots
      self.field_snapshot_divider = field_snapshot_divider
      self.elec_snapshot_divider = elec_snapshot_divider
      self.downsample_rate_lab_x  = downsample_rate_lab_x
      self.downsample_rate_lab_y  = downsample_rate_lab_y
      self.downsample_rate_lab_z  = downsample_rate_lab_z
      self.momentum_resolution    = momentum_resolution
      self.momentum_threshold_low = momentum_threshold_low
      self.momentum_threshold_high= momentum_threshold_high
      self.gamma_threshold_lab    = gamma_threshold_lab
      self.l_transverse_profile = l_transverse_profile
      self.plasma_long_profile = plasma_long_profile
      self.plasma_trans_profile = plasma_trans_profile
      self.window_velocity_lab  = window_velocity_lab
      self.zmin_window_lab = zmin_window_lab
      self.zmax_window_lab = zmax_window_lab
      self.elec = elec
      if l_beam:  self.beam = beam
      self.em   = em
      self.density = density
      self.l_beam = l_beam
      self.runid = runid
      self.l_collapse = l_collapse
      
      if not top.xoldpid:top.xoldpid=nextpid()
      if not top.yoldpid:top.yoldpid=nextpid()
      if not top.zoldpid:top.zoldpid=nextpid()
      if not top.uxoldpid:top.uxoldpid=nextpid()
      if not top.uyoldpid:top.uyoldpid=nextpid()
      if not top.uzoldpid:top.uzoldpid=nextpid()
              
      if me==0:os.system('rm -fr '+output_dir)
      self.create_paths(output_dir)
      self.snapshots = {}
      for datatype in datatypes:
          self.snapshots[datatype] = self.create_lab_snapshots_datatype(n_lab_snapshots, 
                                                         datatypes, 
                                                         datatype, 
                                                         l_restart,
                                                         )

      installafterstep(self.boosted_output_method)  #backtransforms boosted frame data to lab frame and writes output into lab files
      
  def create_lab_snapshots_datatype(self,tn, datatypes, datatype, l_restart):
    """Creates files for backtransformed lab frame data output for a given datatype (elec, field, beam)."""
    
    if w3d.solvergeom == w3d.XYZgeom:dim="3d"
    if w3d.solvergeom in [w3d.XZgeom,w3d.RZgeom]:dim="2d"
    if w3d.solvergeom == w3d.Zgeom:dim="1d"
    
    len_zlab = self.zmax_window_lab-self.zmin_window_lab

    snapshots = [None]*(tn+1)        #Creates empty array for tn snapshots 
        
    for i in numpy.arange(0,tn+1,1): #iterates over snapshots
        
        if me == 0:                  #create files only on process 0
            
            if datatype == 'elec_lab' and i % self.elec_snapshot_divider != 0: continue    #checks if there are less snapshots (elec_snapshot_divider > 1) for elec output and skips iteration
            if datatype == 'field_lab' and i % self.field_snapshot_divider != 0: continue  #checks if there are less snapshots (field_snapshot_divider > 1) for field output and skips iteration

            filepath = os.path.join(self.output_dir, '%s' %(datatype), '%s-%s-%04d.h5' % (self.runid, datatype, i))
            if os.path.exists(filepath) and not l_restart: # this prevents io errors when operating on badly closed h5 files - will remove them
                os.remove(filepath)
            f = h5py.File(filepath, libver="latest") #creates h5 file (libver = latest ensures latest driver version to get max performance)



            snapshots[i] = f #writes the file into the snapshot array at the correct position

            for dataset in datatypes[datatype]: #iterates over all dataset entries in the dictionary "datatype" for the current datatype (elec, field, beam)
                
                if datatype == 'field_lab':
                        #creates datasets for fields for 2d or 3d case. Chunks = True ensures higher performance and less i/o rates.
                        array_shape = (w3d.nx/self.downsample_rate_lab_x+1, 
                                       w3d.ny/self.downsample_rate_lab_y+1, 
                                       int((len_zlab/abs(self.z_lab(top.dt,0,tn)))/self.downsample_rate_lab_z))  #shape of the dataset for the 3d case, longitudinal size = length of the simulation box in lab frame divided by zlab_delta (self.z_lab_(top.dt...))
                        if dim == '2d':
                            shape2d = (int(array_shape[0]), 1, int(array_shape[2]))                         #reduced shape for 2d case
                            dset = snapshots[i].require_dataset(dataset, shape2d, dtype=numpy.float32, chunks=True, exact=False)

                        elif dim == '3d' and fieldslices3d == 1:
                            shape2d = (int(array_shape[0]), 2, int(array_shape[2]))                         #for 3d case and sliced output == True, 2x 2D shape
                            dset = snapshots[i].require_dataset(dataset, shape2d, dtype=numpy.float32, chunks=True, exact=False)
                        else:
                            dset = snapshots[i].require_dataset(dataset, array_shape, dtype=numpy.float32, chunks=True, exact=False) #3d shape

                if datatype == 'elec_lab':
                    #creates datasets for density and phase spaces for 2d or 3d case. Chunks = True ensures higher performance and less i/o rates.
                    array_shape = (w3d.nx/self.downsample_rate_lab_x+1, w3d.ny/self.downsample_rate_lab_y+1, int((len_zlab/abs(self.z_lab(top.dt,0,tn)))/self.downsample_rate_lab_z))
                    
                    if dataset == 'ne':
                        if dim == '2d':
                            shape2d = (int(array_shape[0]), 1, int(array_shape[2]))                         #reduced shape for 2d case
                            dset = snapshots[i].require_dataset(dataset, shape2d, dtype=numpy.float32, chunks=True, exact=False)
                        elif dim == '3d' and densityslices3d == 1:
                            shape2d = (int(array_shape[0]), 2, int(array_shape[2]))                         #for 3d case and sliced output == True, 2x 2D shape
                            dset = snapshots[i].require_dataset(dataset, shape2d, dtype=numpy.float32, chunks=True, exact=False)
                        else:
                            dset = snapshots[i].require_dataset(dataset, array_shape, dtype=numpy.float32, chunks=True, exact=False) #3d shape
                
                    if dataset == 'phase_space' or dataset == 'phase_space_low':
                        shape_phase_space = (self.momentum_resolution, 1, int(array_shape[2])) #shape for phasespace
                        dset = snapshots[i].require_dataset(dataset, shape_phase_space, chunks=True, dtype=numpy.float64, exact=False) 
                
                if datatype == 'beam_lab':
                        #creates datasets for beam files
                        if l_restart:
                          dset = snapshots[i].require_dataset(dataset, snapshots[i][dataset].shape, maxshape=(None,), chunks=True, dtype=numpy.float32, exact=False) #1D resizable array
                        else:
                          dset = snapshots[i].create_dataset(dataset, (0,), maxshape=(None,), chunks=True, dtype=numpy.float32) #1D resizable array

    return snapshots
    
  def boosted_output_method(self):
    """Called after every step. Transforms back data from boosted frame to lab frame and fills lab frame files with slabs of data."""
    
    if w3d.solvergeom == w3d.XYZgeom:dim="3d"
    if w3d.solvergeom in [w3d.XZgeom,w3d.RZgeom]:dim="2d"
    if w3d.solvergeom == w3d.Zgeom:dim="1d"
    
    elec=self.elec
    if self.l_beam:
      beam=self.beam
    em = self.em
    gammafrm = self.gammaframe
    betafrm = self.betaframe
    plasma_long_profile = self.plasma_long_profile
    
    n_lab_snapshots = self.n_lab_snapshots
    snapshots = self.snapshots
    datatypes = self.datatypes
    
    #NOTE: Boosted Output Method uses python map() several times to iterate over all snapshots.

    #dataset_list defines all output datasets available
    dataset_list = {'field_lab': ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'],
                    'elec_lab': ['ne', 'phase_space', 'phase_space_low'],
                    'beam_lab': ['gamma', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'w','t']}

    tn = n_lab_snapshots #number of snapshots defined in warp_run
    t = top.time #time in boosted frame
    
    if not top.it % self.downsample_rate_lab_z == 0: return #if downsamplerate_z > 1 => skips iteration

    i = numpy.array(numpy.arange(0,tn+1,1))     #create array with len(n_lab_snapshots)

    zlab         = self.z_lab(t,i,tn)
    zlabo         = self.z_lab(t-top.dt*self.downsample_rate_lab_z,i,tn)
  
    #get array of accessible lab positions for current timestep and all snapshots
     
    zlab_delta   = abs(self.z_lab(top.dt,0,tn))      #distance that zlab moves every top.dt
   
    zboost       = self.z_boost(t,i,tn)
    zboosto       = self.z_boost(t-top.dt*self.downsample_rate_lab_z,i,tn)#get array of accessible boosted frame positions for current timestep and all snapshots
    zboost_delta = abs(self.z_boost(top.dt,0,tn))    #distance that zboost moves every top.dt
    zboost_prev = self.z_boost(t-top.dt*self.downsample_rate_lab_z,i,tn) # positon of output plane one (downsampled) timestep earlier

    zlab_min     = self.zlab_position(i, tn)[0]      #lower boundary of simulation box if simulation would be in lab frame
    zlab_max     = self.zlab_position(i, tn)[1]      #upper boundary of simulation box if simulation would be in lab frame

    zlab_n       = (zlab - zlab_min)/zlab_delta #index of zlab with respect to the snapshot 
    zlab_n       = zlab_n/self.downsample_rate_lab_z #index correction if donwsamplerate_z > 1

    z_win_min    = w3d.zmmin+top.time*top.vbeamfrm
    zboost_n     = (zboost - z_win_min)/w3d.dz  #index of zboost with respect to simulation box in boosted frame

    access = []  #array to store indices of currently accessible snapshots

    for jjj in numpy.arange(0,tn+1,1): 
      if zlab[jjj] > zlab_min[jjj] and zlab[jjj] < zlab_max[jjj]: #check if currently accessible self.z_lab position lies within boundaries of simulation box in lab frame
        access.append(jjj) #add to access array

    ix       = access
    ix_field = [f_snap for f_snap in ix if f_snap in i[::self.field_snapshot_divider]] #filter ix (access) if there should be less field snapshots 
    ix_elec  = [e_snap for e_snap in ix if e_snap in i[::self.elec_snapshot_divider]]  #filter ix if there should be less elec snapshots
    
    ia       = numpy.arange(len(ix))       #create array of len(ix) for iteration purposes
    ia_field = numpy.arange(len(ix_field)) #same for field snapshots
    ia_elec  = numpy.arange(len(ix_elec))  #same for elec snapshots

    if not ix: return #if ix is empty, return

    #PARTICLES
    #Get particle positions, velocities, gamma and weights for all snapshots currently accessible.
    xa        = elec.getx(gather = 0, bcast = 0)
    ya        = elec.gety(gather = 0, bcast = 0)
    za        = elec.getz(gather = 0, bcast = 0)
    
    uxa       = elec.getux(gather = 0, bcast = 0)
    uya       = elec.getuy(gather = 0, bcast = 0)
    uza       = elec.getuz(gather = 0, bcast = 0)
    
    gammainva = elec.getgaminv(gather = 0, bcast = 0)
    wa        = elec.getweights(gather = 0, bcast = 0)
    
    if self.l_collapse:
        xao        = elec.getpid(id=top.xoldpid-1, gather = 0, bcast = 0)
        yao        = elec.getpid(id=top.yoldpid-1, gather = 0, bcast = 0)
        zao        = elec.getpid(id=top.zoldpid-1, gather = 0, bcast = 0)
        uxao        = elec.getpid(id=top.uxoldpid-1, gather = 0, bcast = 0)
        uyao        = elec.getpid(id=top.uyoldpid-1, gather = 0, bcast = 0)
        uzao        = elec.getpid(id=top.uzoldpid-1, gather = 0, bcast = 0)
    
    z_preva = za - gammainva*uza*top.dt*self.downsample_rate_lab_z
    iia          = numpy.arange(len(za))
    # find particles that crossed the output plane
    ii = map((lambda x: numpy.compress(numpy.logical_or(
                                       numpy.logical_and(numpy.less_equal(zboost[x],za),numpy.less_equal(z_preva,zboost_prev[x])),
                                       numpy.logical_and(numpy.less_equal(za,zboost[x]),numpy.less_equal(zboost_prev[x],z_preva))
                                                        ),iia)), ix)
    
    #create array of len(za) for filtering process
    #Filter particle data and create array with particle data for each snapshot
    
    x        = numpy.array(map(lambda x: xa[ii[x]], ia)) 
    y        = numpy.array(map(lambda x: ya[ii[x]], ia))
    z        = numpy.array(map(lambda x: za[ii[x]], ia))
    w        = numpy.array(map(lambda x: wa[ii[x]], ia))
    ux       = numpy.array(map(lambda x: uxa[ii[x]], ia)) 
    uy       = numpy.array(map(lambda x: uya[ii[x]], ia))
    uz       = numpy.array(map(lambda x: uza[ii[x]], ia))
            
    
    if self.l_collapse:
        
                xo        = numpy.array(map(lambda x: xao[ii[x]], ia)) 
                yo        = numpy.array(map(lambda x: yao[ii[x]], ia))
                zo        = numpy.array(map(lambda x: zao[ii[x]], ia))
                uxo       = numpy.array(map(lambda x: uxao[ii[x]], ia)) 
                uyo       = numpy.array(map(lambda x: uyao[ii[x]], ia))
                uzo       = numpy.array(map(lambda x: uzao[ii[x]], ia))
        
    gammainv = numpy.array(map(lambda x: gammainva[ii[x]], ia))
    gammainvo    = numpy.array(map(lambda x: 1./sqrt(1.+(uxo[x]**2+uyo[x]**2+uzo[x]**2)/clight**2), ia))
    uzfrm    = -betafrm*gammafrm*clight
    #-- Conversion to the lab frame, zo, to, uxo, uyo, uzo
    map(lambda x: setu_in_uzboosted_frame3d(shape(zo[x])[0], 
                                                                                            uxo[x],
                                                                                            uyo[x], 
                                                                                            uzo[x], 
                                                                                            gammainvo[x],
                                                                                            uzfrm, gammafrm), ia) #back transformation of velocities to lab frame
        #Conversion to the lab frame, z, t ux, uy, uz 
    map(lambda x: setu_in_uzboosted_frame3d(shape(z[x])[0], 
                                                                                            ux[x],
                                                                                            uy[x], 
                                                                                            uz[x], 
                                                                                            gammainv[x],
                                                                                            uzfrm, gammafrm), ia) #back transformation of velocities to lab frame
                                                                                            
    t           = numpy.array(map(lambda x: gammafrm*top.time*numpy.ones(shape(z[x])[0])-uzfrm*z[x]/clight**2,ia))
    to         = numpy.array(map(lambda x: gammafrm*(top.time-top.dt)*numpy.ones(shape(z[x])[0])-uzfrm*zo[x]/clight**2,ia))
    z        = numpy.array(map(lambda x: zlab_delta/zboost_delta*(z[x]-zboost[x+ix[0]]) + zlab[x+ix[0]], ia)) #back transformation 
    zo        = numpy.array(map(lambda x: zlab_delta/zboost_delta*(zo[x]-zboosto[x+ix[0]]) + zlabo[x+ix[0]], ia))
    vx       = numpy.array(map(lambda x: ux[x]*gammainv[x], ia))
    vy       = numpy.array(map(lambda x: uy[x]*gammainv[x], ia))
    vz       = numpy.array(map(lambda x: uz[x]*gammainv[x], ia))
    vxo       = numpy.array(map(lambda x: uxo[x]*gammainvo[x], ia))
    vyo       = numpy.array(map(lambda x: uyo[x]*gammainvo[x], ia))
    vzo       = numpy.array(map(lambda x: uzo[x]*gammainvo[x], ia))
    #print "t", t
    #print "to", to
    tlab   = numpy.array(map(lambda x : (numpy.average(t[x]+to[x])/2), ia))
    #print "tlab", tlab
    #Interpolation
        
    if True:
      x      = numpy.array(map(lambda xx : xo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+x[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
      y      = numpy.array(map(lambda xx : yo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+y[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
      vx  = numpy.array(map(lambda xx : vxo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+vx[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
      vy  = numpy.array(map(lambda xx : vyo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+vy[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
      vz  = numpy.array(map(lambda xx : vzo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+vz[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
      gammainv = numpy.array(map(lambda xx : gammainvo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+gammainv[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
      z      = numpy.array(map(lambda xx : zo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+z[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
      t      = numpy.array(map(lambda xx : to[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+t[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
      #print "t_new", t
      #print "***t_lab***", tlab
      gamma    = numpy.array(map(lambda x: sqrt(1.+(ux[x]**2+uy[x]**2+uz[x]**2)/clight**2), ia))
      uzn      = numpy.array(map(lambda x: uz[x]/clight, ia))

    #FIELDS
    if dim == '2d':
      #Get fields with em.gatherXX(). 
      #Fields are directly sliced in z during the gathering process. 
      #Slices are taken at the z_boost position corresponding to the correct self.z_lab position of each snapshot.
      field_data = map(lambda x: [em.gatherex(direction = 2, slice = int(zboost_n[x])), 
                                  em.gatherey(direction = 2, slice = int(zboost_n[x])),
                                  em.gatherez(direction = 2, slice = int(zboost_n[x])),
                                  em.gatherbx(direction = 2, slice = int(zboost_n[x])),
                                  em.gatherby(direction = 2, slice = int(zboost_n[x])),
                                  em.gatherbz(direction = 2, slice = int(zboost_n[x]))], ix_field)

      field_data = numpy.array(field_data)

      if me == 0:
        #Fields are transformed to lab frame
        map(lambda x: seteb_in_boosted_frame(shape(field_data[x,0])[0], 
                                             field_data[x,0], 
                                             field_data[x,1], 
                                             field_data[x,2], 
                                             field_data[x,3], 
                                             field_data[x,4], 
                                             field_data[x,5], 
                                             0., 0., -gammafrm*betafrm*clight, gammafrm), ia_field)
        #write data to array field_data for writing process
        field_data = numpy.array([field_data[:,0], 
                                  field_data[:,1], 
                                  field_data[:,2], 
                                  field_data[:,3], 
                                  field_data[:,4], 
                                  field_data[:,5]])

    else:
      #Get fields with em.gatherXX(). Fields are directly sliced in z during the gathering process. 
      #Slices are taken at the z_boost position corresponding to the correct self.z_lab position of each snapshot. 
      #In 3D case, the arrays are reshaped to 1D arrays.
      field_data = map(lambda x: [numpy.ravel(em.gatherex(direction = 2, slice = int(zboost_n[x]))), 
                                  numpy.ravel(em.gatherey(direction = 2, slice = int(zboost_n[x]))), 
                                  numpy.ravel(em.gatherez(direction = 2, slice = int(zboost_n[x]))), 
                                  numpy.ravel(em.gatherbx(direction = 2, slice = int(zboost_n[x]))), 
                                  numpy.ravel(em.gatherby(direction = 2, slice = int(zboost_n[x]))), 
                                  numpy.ravel(em.gatherbz(direction = 2, slice = int(zboost_n[x])))], ix_field)

      field_data = numpy.array(field_data)

      if me == 0:
        #Fields are transformed to lab frame
        map(lambda x: seteb_in_boosted_frame(shape(field_data[x,0])[0], 
                                             field_data[x,0], 
                                             field_data[x,1], 
                                             field_data[x,2], 
                                             field_data[x,3], 
                                             field_data[x,4], 
                                             field_data[x,5], 
                                             0., 0., -gammafrm*betafrm*clight, gammafrm), ia_field)
        #write data to array field_data for writing process
        field_data = numpy.array([field_data[:,0], 
                                  field_data[:,1], 
                                  field_data[:,2], 
                                  field_data[:,3], 
                                  field_data[:,4], 
                                  field_data[:,5]])

    def get_lab_rho(x):
          """Back transformation of the density to lab frame. Depends on rho and jz in boosted frame."""
          return (1/echarge)*(gammafrm*(rho[x]+jz[x]*(betafrm/clight)))
    #DENSITY
    if dim == '2d':
      #Get rho (charge density) and jz (current density) with em.gatherXX(). 
      #Quantities are directly sliced in z during the gathering process. 
      #Slices are taken at the z_boost position corresponding to the correct self.z_lab position of each snapshot.
      dens = []
      rho  = numpy.array(map(lambda x: numpy.array(em.gatherrho(direction = 2, slice = zboost_n[x])), ix_elec))
      jz   = numpy.array(map(lambda x: numpy.array(em.gatherjz(direction = 2, slice = zboost_n[x])), ix_elec))

      #In order to get the correct back transformed density (calculated from backtransformed charge density) we need to take into account the plasma profile in longitudinal and transverse direction.
      if self.l_transverse_profile == 0:
        if me == 0:
          #Calculates the lab density from the backtransformed charge density and sets correct initial background density with plasma_long_profile.
          dens = numpy.array(map(lambda x: plasma_long_profile(numpy.ones(size(rho[x])), 
                                                               numpy.ones(size(rho[x]))*zlab[x*self.elec_snapshot_divider+ix_elec[0]]/gammafrm)*self.density/gammafrm-get_lab_rho(x), ia_elec))
      
      else:
        #If there is also a transverse density profile we also need to apply plasma_trans_profile.
        if me == 0:
          xm   = numpy.arange(w3d.xmmin, w3d.xmmax, (w3d.xmmax-w3d.xmmin)/(w3d.nx+1)) #Array with transverse positions.
          #Calculates the lab density from the backtransformed charge density and sets correct initial background density with plasma_long_profile and plasma_trans_profile.
          dens = numpy.array(map(lambda x: plasma_trans_profile(plasma_long_profile(numpy.ones(size(rho[x])), 
                                                                                    numpy.ones(size(rho[x]))*zlab[x*self.elec_snapshot_divider+ix_elec[0]]/gammafrm), 
                                                                sqrt(xm**2), 
                                                                numpy.ones(size(rho[x]))*zlab[x*self.elec_snapshot_divider+ix_elec[0]]/gammafrm)*density/gammafrm-get_lab_rho(x), ia_elec))

    else:
      #Get rho (charge density) and jz (current density) with em.gatherXX(). 
      #Quantities are directly sliced in z during the gathering process. 
      #Slices are taken at the z_boost position corresponding to the correct self.z_lab position of each snapshot.
      #In 3D case, the arrays are reshaped to 1D arrays.
      dens = []
      rho  = numpy.array(map(lambda x: numpy.array(numpy.ravel(em.gatherrho(direction = 2, slice = zboost_n[x]))), ix_elec))
      jz   = numpy.array(map(lambda x: numpy.array(numpy.ravel(em.gatherjz(direction = 2, slice = zboost_n[x]))), ix_elec))

      #In order to get the correct back transformed density (calculated from backtransformed charge density) we need to take into account the plasma profile in longitudinal and transverse direction.
      if self.l_transverse_profile == 0:
        if me == 0:
          #Calculates the lab density from the backtransformed charge density and sets correct initial background density with plasma_long_profile.
          dens   = numpy.array(map(lambda x: plasma_long_profile(numpy.ones(size(rho[x])), 
                                                               numpy.ones(size(rho[x]))*zlab[x*self.elec_snapshot_divider+ix_elec[0]]/gammafrm)*density/gammafrm-get_lab_rho(x), ia_elec))
      
      else:
        if me == 0:
          #If there is also a transverse density profile we also need to apply plasma_trans_profile.
          xm, ym = meshgrid(numpy.arange(w3d.xmmin,w3d.xmmax,(w3d.xmmax-w3d.xmmin)/(w3d.nx+1)), # 1D Arrays with transverse positions.
                            numpy.arange(w3d.ymmin,w3d.ymmax,(w3d.ymmax-w3d.ymmin)/(w3d.ny+1)))
          #Calculates the lab density from the backtransformed charge density and sets correct initial background density with plasma_long_profile and plasma_trans_profile.
          dens   = numpy.array(map(lambda x: plasma_trans_profile(plasma_long_profile(numpy.ones(size(rho[x])), 
                                                                                      numpy.ones(size(rho[x]))*zlab[x*self.elec_snapshot_divider+ix_elec[0]]/gammafrm),
                                                                  numpy.ravel(sqrt(xm**2+ym**2)), 
                                                                  numpy.ones(size(rho[x]))*zlab[x*self.elec_snapshot_divider+ix_elec[0]]/gammafrm)*density/gammafrm-get_lab_rho(x), ia_elec))
      
    #PHASESPACEHIGH
    #Get phasespace by taking the parallel sum of a histogram of uzn at the current self.z_lab position of each snapshot.
    phasespace    = map(lambda x: parallelsum(numpy.histogram(uzn[x*self.elec_snapshot_divider], 
                                                              bins = self.momentum_resolution,
                                                              range = (-self.momentum_threshold_high, self.momentum_threshold_high),
                                                              weights = w[x*self.elec_snapshot_divider])[0]), ia_elec)
    
    #PHASESPACELOW
    #Get phasespace by taking the parallel sum of a histogram of uzn at the current self.z_lab position of each snapshot.
    phasespacelow = map(lambda x: parallelsum(numpy.histogram(uzn[x*self.elec_snapshot_divider], 
                                                              bins = self.momentum_resolution,
                                                              range = (-self.momentum_threshold_low, self.momentum_threshold_low),
                                                              weights = w[x*self.elec_snapshot_divider])[0]), ia_elec)

    elec_data = [dens, phasespace, phasespacelow] #write data to array elec_data for writing process

    if self.l_beam == 0:

      jj    = numpy.array(map((lambda x: numpy.compress(numpy.less(self.gamma_threshold_lab,gamma[x]),arange(len(ii[x])))), ia)) #filter particles indices by gamma_threshold

      #FILTER AND GATHER BEAM (no external beam)
      #Select particles with gamma > gamma_threshold_lab and gather them on process 0
      x     = map(lambda xx: numpy.array(gatherarray(x[xx][jj[xx]])), ia)
      y     = map(lambda x: numpy.array(gatherarray(y[x][jj[x]])), ia)
      z     = map(lambda x: numpy.array(gatherarray(z[x][jj[x]])), ia)

      w     = map(lambda x: numpy.array(gatherarray(w[x][jj[x]])), ia)

      vx    = map(lambda x: numpy.array(gatherarray(vx[x][jj[x]])), ia)
      vy    = map(lambda x: numpy.array(gatherarray(vy[x][jj[x]])), ia)
      vz    = map(lambda x: numpy.array(gatherarray(vz[x][jj[x]])), ia)

      gamma = map(lambda x: numpy.array(gatherarray(gamma[x][jj[x]])), ia)
      
      
    #EXTERNAL BEAM
    if self.l_beam == 1:

      #Get beam particle positions, velocities, gamma and weights for all snapshots currently accessible.
      xb        = beam.getx(gather = 0, bcast = 0).copy()
      yb        = beam.gety(gather = 0, bcast = 0).copy()
      zb        = beam.getz(gather = 0, bcast = 0).copy()
      
      uxb       = beam.getux(gather = 0, bcast = 0).copy()
      uyb       = beam.getuy(gather = 0, bcast = 0).copy()
      uzb       = beam.getuz(gather = 0, bcast = 0).copy()
      
      gammainvb = beam.getgaminv(gather = 0, bcast = 0)
      wb        = beam.getweights(gather = 0, bcast = 0)
      
      if self.l_collapse:
                    xbo        = beam.getpid(id=top.xoldpid-1, gather = 0, bcast = 0)
                    ybo        = beam.getpid(id=top.yoldpid-1, gather = 0, bcast = 0)
                    zbo        = beam.getpid(id=top.zoldpid-1, gather = 0, bcast = 0)
                    uxbo        = beam.getpid(id=top.uxoldpid-1, gather = 0, bcast = 0)
                    uybo        = beam.getpid(id=top.uyoldpid-1, gather = 0, bcast = 0)
                    uzbo        = beam.getpid(id=top.uzoldpid-1, gather = 0, bcast = 0)    
      iib = numpy.arange(len(zb))
      # calculate z positions one timestep earlier
      #z_prevb = zb - gammainvb*uzb*top.dt*self.downsample_rate_lab_z
      #ii = map((lambda x: numpy.compress(numpy.logical_and(numpy.less_equal(zboost[x],zb),numpy.less_equal(z_prevb,zboost_prev[x])),iib)), ix)

      # find particles that crossed the output plane
      ii = map((lambda x: numpy.compress(numpy.logical_or(numpy.logical_and(numpy.less_equal(zboost[x],zb),numpy.less_equal(zbo,zboost_prev[x])),numpy.logical_and(numpy.less_equal(zb,zboost[x]),numpy.less_equal(zboost_prev[x],zbo))),iib)), ix)
        
      #Filter particle data and create array with particle data for each snapshot
      x        = numpy.array(map(lambda x: xb[ii[x]], ia)) 
      y        = numpy.array(map(lambda x: yb[ii[x]], ia))
      z        = numpy.array(map(lambda x: zb[ii[x]], ia))

      ux       = numpy.array(map(lambda x: uxb[ii[x]], ia)) 
      uy       = numpy.array(map(lambda x: uyb[ii[x]], ia))
      uz       = numpy.array(map(lambda x: uzb[ii[x]], ia))
    
      if self.l_collapse:
        
        xo        = numpy.array(map(lambda x: xbo[ii[x]], ia)) 
        yo        = numpy.array(map(lambda x: ybo[ii[x]], ia))
        zo        = numpy.array(map(lambda x: zbo[ii[x]], ia))
        uxo       = numpy.array(map(lambda x: uxbo[ii[x]], ia)) 
        uyo       = numpy.array(map(lambda x: uybo[ii[x]], ia))
        uzo       = numpy.array(map(lambda x: uzbo[ii[x]], ia))
                
      gammainv = numpy.array(map(lambda x: gammainvb[ii[x]], ia))
      gammainvo    = numpy.array(map(lambda x: 1./sqrt(1.+(uxo[x]**2+uyo[x]**2+uzo[x]**2)/clight**2), ia))

      w        = numpy.array(map(lambda x: wb[ii[x]], ia))
      uzfrm    = -betafrm*gammafrm*clight
        
      #-- Conversion to the lab frame, zo, to, uxo, uyo, uzo
      map(lambda x: setu_in_uzboosted_frame3d(shape(zo[x])[0], 
                                                                                            uxo[x],
                                                                                            uyo[x], 
                                                                                            uzo[x], 
                                                                                            gammainvo[x],
                                                                                            uzfrm, gammafrm), ia) #back transformation of velocities to lab frame
     
      #Conversion to the lab frame, z, t ux, uy, uz 
      map(lambda x: setu_in_uzboosted_frame3d(shape(z[x])[0], 
                                                                                            ux[x],
                                                                                            uy[x], 
                                                                                            uz[x], 
                                                                                            gammainv[x],
                                                                                            uzfrm, gammafrm), ia) #back transformation of velocities to lab frame
     
      t           = numpy.array(map(lambda x: gammafrm*top.time*numpy.ones(shape(z[x])[0])-uzfrm*z[x]/clight**2,ia))
      to         = numpy.array(map(lambda x: gammafrm*(top.time-top.dt)*numpy.ones(shape(z[x])[0])-uzfrm*zo[x]/clight**2,ia))
      z        = numpy.array(map(lambda x: zlab_delta/zboost_delta*(z[x]-zboost[x+ix[0]]) + zlab[x+ix[0]], ia)) #back transformation 
      zo        = numpy.array(map(lambda x: zlab_delta/zboost_delta*(zo[x]-zboosto[x+ix[0]]) + zlabo[x+ix[0]], ia))
        
      vx       = numpy.array(map(lambda x: ux[x]*gammainv[x], ia))
      vy       = numpy.array(map(lambda x: uy[x]*gammainv[x], ia))
      vz       = numpy.array(map(lambda x: uz[x]*gammainv[x], ia))
      vxo       = numpy.array(map(lambda x: uxo[x]*gammainvo[x], ia))
      vyo       = numpy.array(map(lambda x: uyo[x]*gammainvo[x], ia))
      vzo       = numpy.array(map(lambda x: uzo[x]*gammainvo[x], ia))
      #print "t", t
      #print "to", to
      tlab   = numpy.array(map(lambda x : (numpy.average(t[x]+to[x])/2), ia))
      #print "tlab", tlab
        
      #Interpolation
      if True:
        x      = numpy.array(map(lambda xx : xo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+x[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
        y      = numpy.array(map(lambda xx : yo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+y[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
        vx  = numpy.array(map(lambda xx : vxo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+vx[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
        vy  = numpy.array(map(lambda xx : vyo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+vy[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
        vz  = numpy.array(map(lambda xx : vzo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+vz[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
        gammainv = numpy.array(map(lambda xx : gammainvo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+gammainv[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
        z      = numpy.array(map(lambda xx : zo[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+z[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))
        t      = numpy.array(map(lambda xx : to[xx]*(t[xx]-tlab[xx])/(t[xx]-to[xx])+t[xx]*(tlab[xx]-to[xx])/(t[xx]-to[xx]),ia))

        
                #print "t_new", t
                #print "***t_lab***", tlab
            
        gamma    = numpy.array(map(lambda x: sqrt(1.+(ux[x]**2+uy[x]**2+uz[x]**2)/clight**2), ia))
        uzn      = numpy.array(map(lambda x: uz[x]/clight, ia))

        #Gather
        x     = map(lambda xx: numpy.array(gatherarray(x[xx])), ia)
        y     = map(lambda x: numpy.array(gatherarray(y[x])), ia)
        z     = map(lambda x: numpy.array(gatherarray(z[x])), ia)

        w     = map(lambda x: numpy.array(gatherarray(w[x])), ia)

        vx    = map(lambda x: numpy.array(gatherarray(vx[x])), ia)
        vy    = map(lambda x: numpy.array(gatherarray(vy[x])), ia)
        vz    = map(lambda x: numpy.array(gatherarray(vz[x])), ia)

        gamma = map(lambda x: numpy.array(gatherarray(gamma[x])), ia)


    beam_data = [gamma, x, y, z, vx, vy, vz, w, t] #write data to array beam_data for writing process

    def write_data(x):
      #Mappable function which writes the lab data slices to the corresponding position of the h5 datasets.
      if me == 0:
        #Checks if current self.z_lab position lies within boundaries of the virtual lab frame simulation box. 
        if zlab[x] > zlab_min[x] and zlab[x] < zlab_max[x]: 

          if 'field_lab' in datatypes and x in ix_field:
            
            f = snapshots['field_lab'][x] #get snapshot file

            for dataset in datatypes['field_lab']:
              #iterates over all datasets defined in datatypes.
              dset = f['%s' %(dataset)] #open dataset
              #Writes fields
              if dim == '2d':
                  #2D Output
                  dset[:,0,zlab_n[x]-1] = field_data[dataset_list['field_lab'].index(dataset)][(x-ix_field[0])/self.field_snapshot_divider][::self.downsample_rate_lab_x]
              else:
                  if fieldslices3d == 1:
                      #Sliced Output. Writes only the on-axis slices in x and y direction to the file.
                      dset[:,0,zlab_n[x]-1] = reshape(field_data[dataset_list['field_lab'].index(dataset)][(x-ix_field[0])/self.field_snapshot_divider],(w3d.nx+1, w3d.ny+1),'F')[::self.downsample_rate_lab_x,int(w3d.ny+1/2)]
                      dset[:,1,zlab_n[x]-1] = reshape(field_data[dataset_list['field_lab'].index(dataset)][(x-ix_field[0])/self.field_snapshot_divider],(w3d.nx+1, w3d.ny+1),'F')[int(w3d.nx+1/2),::self.downsample_rate_lab_y]
                  else:
                      #Full 3D Output
                      dset[:,:,zlab_n[x]-1] = reshape(field_data[dataset_list['field_lab'].index(dataset)][(x-ix_field[0])/self.field_snapshot_divider],(w3d.nx+1, w3d.ny+1),'F')[::self.downsample_rate_lab_x,::self.downsample_rate_lab_y]
          
          if 'elec_lab' in datatypes and x in ix_elec:
                        
            f = snapshots['elec_lab'][x] #get snapshot file

            for dataset in datatypes['elec_lab']:
                #iterates over all datasets defined in datatypes.
                dset = f['%s' %(dataset)] #open dataset
                #Write density
                if dataset == 'ne':
                    if dim == '2d':
                        #2D Output
                        dset[:,0,zlab_n[x]-1] = elec_data[dataset_list['elec_lab'].index(dataset)][(x-ix_elec[0])/self.elec_snapshot_divider][::self.downsample_rate_lab_x]
                    else:
                        if densityslices3d == 1:
                            #Sliced Output
                            dset[:,0,zlab_n[x]-1] = reshape(elec_data[dataset_list['elec_lab'].index(dataset)][(x-ix_elec[0])/self.elec_snapshot_divider],(w3d.nx+1, w3d.ny+1),'F')[::self.downsample_rate_lab_x,int(w3d.ny+1/2)]
                            dset[:,1,zlab_n[x]-1] = reshape(elec_data[dataset_list['elec_lab'].index(dataset)][(x-ix_elec[0])/self.elec_snapshot_divider],(w3d.nx+1, w3d.ny+1),'F')[int(w3d.nx+1/2),::self.downsample_rate_lab_y]
                        else:
                            #Full 3D Output
                            dset[:,:,zlab_n[x]-1] = reshape(elec_data[dataset_list['elec_lab'].index(dataset)][(x-ix_elec[0])/self.elec_snapshot_divider],(w3d.nx+1, w3d.ny+1),'F')[::self.downsample_rate_lab_x,::self.downsample_rate_lab_y]
                #Write phasespaces
                if dataset == 'phase_space' or dataset =='phase_space_low':
                    dset[:,0,zlab_n[x]-1] = elec_data[dataset_list['elec_lab'].index(dataset)][(x-ix_elec[0])/self.elec_snapshot_divider]

          if 'beam_lab' in datatypes:
                      
            f = snapshots['beam_lab'][x] #get snapshot file

            for dataset in datatypes['beam_lab']:
                #iterates over all datasets defined in datatypes.
                if size(beam_data[dataset_list['beam_lab'].index(dataset)][x-ix[0]]) != 0: #checks if there are new beam particles to write to 1D h5 dataset.

                    dset      = f['%s' %(dataset)] #open dataset
                    dset_rows = dset.shape[0]
                    dset_data = beam_data[dataset_list['beam_lab'].index(dataset)][x-ix[0]]

                    new_rows  = len(dset_data) + dset.shape[0]
                    #dset.reshape((new_rows,1)) #reshape beam dataset
                    dset.resize(new_rows, axis = 0) #reshape beam dataset
                    dset[dset_rows::] = dset_data #Write beam

    map(write_data,ix) #Iterates write_data.

  def create_paths(self,output_dir):
    ### CHECK OUTPUT DIRECTORY
    self.output_dir = output_dir
    if (not os.path.exists(output_dir)) and me==0:
      print 'Directory for output: ', output_dir, ' does not exists.'
      print 'Trying to create it... '
      os.mkdir(output_dir)
      os.mkdir(os.path.join(output_dir, 'beam'))
      os.mkdir(os.path.join(output_dir, 'elec'))
      os.mkdir(os.path.join(output_dir, 'field'))
      os.mkdir(os.path.join(output_dir, 'field_lab'))
      os.mkdir(os.path.join(output_dir, 'elec_lab'))
      os.mkdir(os.path.join(output_dir, 'beam_lab'))
      if not os.path.exists(output_dir):
        print >> sys.stderr, 'Cannot create the directory'
        print >> sys.stderr, 'Bye, bye!'
        sys.exit()

  def z_lab(self, t, i, tn):
    """Find accessible z-position in lab frame at given diagnostic time in lab frame for given time in boosted frame."""

    return (clight*self.gammaframe*i*self.simulation_time_lab/tn - clight*t)/(self.gammaframe*self.betaframe)


  def z_boost(self, t, i, tn):
    """Find matching z-position (contains data for z-position in lab frame given by z_lab) in boosted frame at given diagnostic time in lab frame for given time in boosted frame."""

    return (clight*i*self.simulation_time_lab/tn - self.gammaframe*clight*t)/(self.gammaframe*self.betaframe)


  def zlab_position(self, i,tn):
    """Moving window boundaries in lab_frame. Virtual lab frame simulation box."""

    zmin = self.window_velocity_lab*i*self.simulation_time_lab/tn + self.zmin_window_lab
    zmax = self.window_velocity_lab*i*self.simulation_time_lab/tn + self.zmax_window_lab
    
    return zmin, zmax
    
  def time_of_snapshot_lab(self,i,tn):
      return i*self.simulation_time_lab/tn
        
  def set_common_attrs_snapshots(self,h5file,i,tn,datatype):
   """
   Stores in h5file some values that can be useful during data visualization
   These parameters here are lab frame values calculated for the lab frame dump in boosted frame simulations
   """
  
   for dataset in h5file:

    # one can use also syntax: str(dataset)
    h5file['%s' %(dataset)].attrs['xmin'] = w3d.xmmin
    h5file['%s' %(dataset)].attrs['ymin'] = w3d.ymmin
    h5file['%s' %(dataset)].attrs['zmin'] = self.zlab_position(i,tn)[0]
    h5file['%s' %(dataset)].attrs['xmax'] = w3d.xmmax
    h5file['%s' %(dataset)].attrs['ymax'] = w3d.ymmax
    h5file['%s' %(dataset)].attrs['zmax'] = self.zlab_position(i,tn)[1]
    
    h5file['%s' %(dataset)].attrs['runid']  = self.runid
    h5file['%s' %(dataset)].attrs['time']   = i*self.simulation_time_lab/tn
    h5file['%s' %(dataset)].attrs['dt']     = top.dt
    h5file['%s' %(dataset)].attrs['diagno'] = i

    if w3d.solvergeom is w3d.XZgeom:
      h5file['%s' %(dataset)].attrs['solvergeom'] = 'XZgeom'
    else:
      h5file['%s' %(dataset)].attrs['solvergeom'] = 'XYZgeom'

    if dataset == 'phase_space':
      h5file['%s' %(dataset)].attrs['phase_space_min'] = -self.momentum_threshold_high
      h5file['%s' %(dataset)].attrs['phase_space_max'] = self.momentum_threshold_high

    if dataset == 'phase_space_low':
      h5file['%s' %(dataset)].attrs['phase_space_min'] = -self.momentum_threshold_low
      h5file['%s' %(dataset)].attrs['phase_space_max'] = self.momentum_threshold_low 

    if datatype == 'beam_lab':
      h5file['%s' %(dataset)].attrs['gamma_threshold'] = self.gamma_threshold_lab


  def close_snapshots(self):
   """
   close files that contain back-transformed labframe data,
   but first set a few attributes
   """

   tn = self.n_lab_snapshots

   if me == 0:  
  
    for i in numpy.arange(0,tn+1,1):

      for datatype in self.datatypes:

        if datatype == 'elec_lab' and i % self.elec_snapshot_divider != 0: continue
        if datatype == 'field_lab' and i % self.field_snapshot_divider != 0: continue
                
        f = self.snapshots[datatype][i]
        self.set_common_attrs_snapshots(f,i,tn,datatype)
        f.close()              
