<template>
  <div class="dtp">
    <button
      :id="id"
      class="dtp-trigger dtp-input"
      :data-open="open ? '1' : '0'"
      data-hour-item-h="32"
      data-minute-item-h="32"
      @click="toggle()"
      @keydown.enter.prevent="toggle()"
    >
      <span class="dtp-display">{{ display }}</span>
      <svg class="dtp-cal-icon" viewBox="0 0 24 24" aria-hidden="true">
        <rect x="3" y="5" width="18" height="16" rx="2" ry="2" stroke="currentColor" fill="none"/>
        <line x1="16" y1="3" x2="16" y2="7" stroke="currentColor"/>
        <line x1="8" y1="3" x2="8" y2="7" stroke="currentColor"/>
        <line x1="3" y1="11" x2="21" y2="11" stroke="currentColor"/>
      </svg>
    </button>

    <div v-if="open" class="dtp-popover" ref="popover" tabindex="0" @keydown.enter.prevent="open=false">
      <div class="dtp-left">
        <div class="dtp-header">
          <div class="ym-vertical" @click="panel = 'month'">{{ monthLabelEn }} {{ y }}</div>
          <div class="dtp-nav">
            <span class="dtp-arrow" @click="prevMonth()">▲</span>
            <span class="dtp-arrow" @click="nextMonth()">▼</span>
          </div>
        </div>

        <div class="dtp-calendar">
          <div v-if="panel === 'month'" class="dtp-title" ref="yearList">
            <div v-for="yy in yearRange" :key="yy" :class="['dtp-year-group', `year-${yy}`]">
              <div :class="['dtp-year-head', { sel: yy === y }]" @click="expandYear(yy)">{{ yy }}</div>
              <div v-show="expandedYear === yy" class="dtp-months">
                <div
                  v-for="mth in 12"
                  :key="mth"
                  class="dtp-month"
                  :class="[`month-${mth}`, { sel: (yy===y && mth === m) }]"
                  @click="chooseMonthForYear(yy, mth)"
                >
                  {{ ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][mth-1] }}
                </div>
              </div>
            </div>
          </div>

          <div v-else>
            <div class="dtp-days">
              <div
                v-for="dd in daysInMonth"
                :key="dd"
                class="dtp-day"
                :class="[`day-${dd}`, { sel: dd === d }]"
                @click="chooseDay(dd)"
              >
                {{ dd }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="dtp-right">
        <!-- Time wheels -->
        <div class="dtp-timecol">
          <div class="dtp-timecol-title">Hour</div>
          <div class="dtp-timecol-list" :id="`${id}-hour`">
            <div
              v-for="hh in 24"
              :key="hh"
              class="dtp-time-item"
              :class="[`hour-${hh-1}`, { sel: (hh-1) === h }]"
              @click="chooseHour(hh-1)"
            >
              {{ (hh-1).toString().padStart(2,'0') }}
            </div>
          </div>
        </div>
        <div class="dtp-timecol">
          <div class="dtp-timecol-title">Minute</div>
          <div class="dtp-timecol-list" :id="`${id}-minute`">
            <div
              v-for="mi in 60"
              :key="mi"
              class="dtp-time-item"
              :class="[`minute-${mi-1}`, { sel: (mi-1) === min }]"
              @click="chooseMinute(mi-1)"
            >
              {{ (mi-1).toString().padStart(2,'0') }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { nextTick } from 'vue'

export default {
  name: 'DateTimePicker',
  props: {
    id: { type: String, default: 'date-picker' },
    modelValue: { type: String, default: '2025-10-15 12:30' }, // 'YYYY-MM-DD HH:mm'
  },
  emits: ['update:modelValue'],
  data() {
    const parsed = this._parse(this.modelValue)
    return {
      open: false,
      panel: 'date',
      expandedYear: parsed.y,
      y: parsed.y,
      m: parsed.m,
      d: parsed.d,
      h: parsed.h,
      min: parsed.min,
    }
  },
  computed: {
    display() {
      const pad = (n) => String(n).padStart(2,'0')
      return `${this.y}-${pad(this.m)}-${pad(this.d)} ${pad(this.h)}:${pad(this.min)}`
    },
    monthLabelEn() {
      const m = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
      return m[this.m-1]
    },
    yearRange() {
      const base = this.y || new Date().getFullYear()
      const arr = []
      for (let yy = base - 4; yy <= base + 5; yy++) arr.push(yy)
      return arr
    },
    daysInMonth() {
      const y = this.y, m = this.m
      if (!y || !m) return []
      return new Date(y, m, 0).getDate()
    }
  },
  watch: {
    modelValue(v) {
      const p = this._parse(v)
      this.y = p.y; this.m = p.m; this.d = p.d; this.h = p.h; this.min = p.min
    }
  },
  methods: {
    toggle() {
      this.open = !this.open
      if (this.open) {
        this.$nextTick(() => {
          const p = this.$refs.popover
          if (p && p.focus) p.focus()

          // 自动滚动到选中的小时和分钟
          this.scrollToSelectedTime()
        })
      }
    },
    scrollToSelectedTime() {
      // 滚动小时选择器
      const hourContainer = document.getElementById(`${this.id}-hour`)
      if (hourContainer) {
        const selectedHourElement = hourContainer.querySelector(`.hour-${this.h}`)
        if (selectedHourElement) {
          selectedHourElement.scrollIntoView({ block: 'center', behavior: 'smooth' })
        }
      }

      // 滚动分钟选择器
      const minuteContainer = document.getElementById(`${this.id}-minute`)
      if (minuteContainer) {
        const selectedMinuteElement = minuteContainer.querySelector(`.minute-${this.min}`)
        if (selectedMinuteElement) {
          selectedMinuteElement.scrollIntoView({ block: 'center', behavior: 'smooth' })
        }
      }
    },
    chooseYear(yy) {
      this.y = yy
      this._emit()
    },
    chooseMonth(mm) {
      this.m = mm
      const dim = new Date(this.y, this.m, 0).getDate()
      if (this.d > dim) this.d = dim
      this.panel = 'date'
      this._emit()
    },
    expandYear(yy) {
      this.y = yy
      this.expandedYear = yy
      this.$nextTick(() => {
        const el = this.$el.querySelector(`.year-${yy}`)
        if (el && el.scrollIntoView) el.scrollIntoView({ block: 'nearest' })
      })
    },
    chooseMonthForYear(yy, mm) {
      this.y = yy
      this.chooseMonth(mm)
    },
    chooseDay(dd) {
      this.d = dd
      this._emit()
    },
    chooseHour(hh) {
      this.h = hh
      this._emit()
    },
    chooseMinute(mi) {
      this.min = mi
      this._emit()
    },
    prevMonth() {
      let y = this.y, m = this.m - 1
      if (m < 1) { m = 12; y -= 1 }
      this.y = y; this.m = m
      const dim = new Date(y, m, 0).getDate()
      if (this.d > dim) this.d = dim
      this._emit()
    },
    nextMonth() {
      let y = this.y, m = this.m + 1
      if (m > 12) { m = 1; y += 1 }
      this.y = y; this.m = m
      const dim = new Date(y, m, 0).getDate()
      if (this.d > dim) this.d = dim
      this._emit()
    },

    _emit() {
      const pad = (n) => String(n).padStart(2,'0')
      const s = `${this.y}-${pad(this.m)}-${pad(this.d)} ${pad(this.h)}:${pad(this.min)}`
      this.$emit('update:modelValue', s)
    },
    _parse(s) {
      const now = new Date()
      if (!s || typeof s !== 'string') {
        const pad = (n) => String(n).padStart(2,'0')
        return {
          y: now.getFullYear(),
          m: now.getMonth() + 1,
          d: now.getDate(),
          h: 0,
          min: 0,
          str: `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())} 00:00`
        }
      }
      // Expect 'YYYY-MM-DD HH:mm' (tolerate 'YYYY-MM-DD')
      try {
        const [datePart, timePart='00:00'] = s.replace('T',' ').split(/\s+/)
        const [Y,M,D] = datePart.split('-').map((x) => parseInt(x,10))
        const [h=0, m=0] = timePart.split(':').map((x) => parseInt(x,10))
        return { y:Y||now.getFullYear(), m:M||1, d:D||1, h: h||0, min: m||0 }
      } catch (e) {
        return { y: now.getFullYear(), m: now.getMonth()+1, d: now.getDate(), h: 0, min: 0 }
      }
    }
  }
}
</script>

<style scoped>
.dtp { display: inline-block; position: relative; }
.dtp-trigger { padding: 6px 10px; border: 1px solid #d1d5db; border-radius: 6px; background: white; cursor: pointer; }
.dtp-input { display: inline-flex; align-items: center; position: relative; padding: 6px 34px 6px 10px; height: 36px; }
.dtp-display { line-height: 1; }
.dtp-cal-icon { position: absolute; right: 10px; width: 16px; height: 16px; color: #9ca3af; }
.dtp-input:hover .dtp-cal-icon { color: #6b7280; }

.dtp-popover { position: absolute; margin-top: 8px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; display: flex; gap: 16px; box-shadow: 0 10px 20px rgba(0,0,0,0.08); z-index: 50; }
.dtp-left { display: grid; grid-template-columns: 1fr; gap: 10px; width: 300px; }
.dtp-title { border: 1px solid #f3f4f6; border-radius: 6px; padding: 4px; }
.dtp-calendar { height: 260px; overflow-y: auto; }

.dtp-year-row { padding: 6px 8px; border-radius: 4px; background: #f9fafb; margin: 2px 0; cursor: pointer; }
.dtp-year-row.sel { background: #e5e7eb; }
.dtp-year-head { padding: 8px 10px; background: #f3f4f6; border-radius: 4px; margin: 6px 0; cursor: pointer; }
.dtp-year-head.sel { background: #e5e7eb; }


.dtp-header { display: flex; align-items: center; justify-content: space-between; }
.ym-verticaw { font-weight: 600; cursor: pointer; }
.dtp-nav { display: flex; gap: 8px; }
.dtp-arrow { cursor: pointer; user-select: none; padding: 2px 6px; border: none; border-radius: 4px; font-size: 12px; background: #fff; }
.dtp-day:hover { background: #f3f4f6; }

.dtp-arrow:hover { background: #e5e7eb; }

.dtp-months { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }

.dtp-month { padding: 6px; text-align: center; border-radius: 4px; background: #f9fafb; cursor: pointer; }
.dtp-month.sel { background: #2563eb; color: white; }
.dtp-days { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; max-width: 280px; }
.dtp-day { padding: 6px; text-align: center; border-radius: 4px; background: transparent; border: none; cursor: pointer; }
.dtp-day.sel { background: #2563eb; color: #fff; }
.dtp-right { display: flex; gap: 12px; }
.dtp-timecol { width: 80px; }
.dtp-timecol-title { font-size: 12px; color: #6b7280; margin-bottom: 6px; }
.dtp-timecol-list { height: 260px; overflow: auto; border: none; border-radius: 6px; padding: 4px; }
.dtp-time-item { height: 32px; line-height: 32px; text-align: center; border-radius: 4px; margin: 2px 0; background: #f9fafb; cursor: pointer; }
.dtp-time-item.sel { background: #2563eb; color: #fff; }
</style>

