#ifndef LV_EVENT_H
#define LV_EVENT_H

///Simple event structure
class Event{
  public:
    // public members
    double energy_proxy;
    double energy_neutrino;
    double costh;
    double conv_kaon_event;
    double conv_pion_event;
    double prompt_event;
    double astro_event;
    unsigned int year;

    Event():
      energy_proxy(std::numeric_limits<double>::quiet_NaN()),
      energy_neutrino(std::numeric_limits<double>::quiet_NaN()),
      costh(std::numeric_limits<double>::quiet_NaN()),
      conv_pion_event(std::numeric_limits<double>::quiet_NaN()),
      conv_kaon_event(std::numeric_limits<double>::quiet_NaN()),
      prompt_event(std::numeric_limits<double>::quiet_NaN()),
      astro_event(std::numeric_limits<double>::quiet_NaN()),
      year(0)
    {}

    Event(double energy_proxy,double costh,unsigned int year):
      energy_proxy(energy_proxy),
      energy_neutrino(std::numeric_limits<double>::quiet_NaN()),
      costh(costh),
      conv_pion_event(std::numeric_limits<double>::quiet_NaN()),
      conv_kaon_event(std::numeric_limits<double>::quiet_NaN()),
      prompt_event(std::numeric_limits<double>::quiet_NaN()),
      astro_event(std::numeric_limits<double>::quiet_NaN()),
      year(year)
    {}

    Event(double energy_proxy,double costh,unsigned int year, double conv_kaon_event, double conv_pion_event):
      energy_proxy(energy_proxy),
      energy_neutrino(std::numeric_limits<double>::quiet_NaN()),
      costh(costh),
      conv_pion_event(conv_pion_event),
      conv_kaon_event(conv_kaon_event),
      prompt_event(0),
      astro_event(0),
      year(year)
    {}
};

#endif
